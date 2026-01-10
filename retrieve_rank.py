# retrieve.py / rank.py
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import BucketedRandomProjectionLSH

def retrieve(target_id, country, df, lsh_model, n=50):
    """ANN retrieval inside one country using precomputed features_norm."""
    q = df.filter(F.col("property_id") == target_id).select("features_norm").limit(1).collect()
    if not q: return None
    q_vec = q[0]["features_norm"]
    cand = df.filter(F.col("addr_cc") == country)
    return lsh_model.approxNearestNeighbors(cand, q_vec, n, distCol="l2_dist")


def _normalize_weights(w: dict):
    """Normalize positive weights to sum to 1."""
    if not w: return {}
    s = sum(v for v in w.values() if v is not None and v > 0)
    if s <= 0: return {}
    return {k: float(v) / s for k, v in w.items() if v is not None and v > 0}


def order(
    df: DataFrame,
    k: int,
    price_w: float = 0.0,
    property_w: float = 0.0,
    host_w: float = 0.0,
    env_weights: dict = None,        # {"env_food": 20, "env_nature": 10, ...} OR {"env_food_norm": 20, ...}
    temp_pref: float = None,         # preferred temperature
    temp_w: float = 0.0,
    travel_month: int = None,        # 1..12
    budget_pref: str = None,         # "Budget" | "Mid-range" | "Luxury"
    budget_w: float = 0.0,
    normalize_all_weights: bool = True,
    score_col: str = "final_score"
) -> DataFrame:
    """Rank an already-filtered DF using precomputed score components + user weights."""
    work = df

    # weights (optionally normalize everything together)
    env_weights = env_weights or {}
    all_w = {"price": price_w, "property": property_w, "host": host_w, "temp": temp_w, "budget": budget_w, **{f"env::{k}": v for k, v in env_weights.items()}}
    w_norm = _normalize_weights(all_w) if normalize_all_weights else all_w
    price_w = w_norm.get("price", 0.0); property_w = w_norm.get("property", 0.0); host_w = w_norm.get("host", 0.0); temp_w = w_norm.get("temp", 0.0); budget_w = w_norm.get("budget", 0.0)

    terms = []

    # base components from airbnb_scores()
    if price_w and "price_score" in work.columns: terms.append(F.coalesce(F.col("price_score"), F.lit(0.0)) * F.lit(price_w))
    if property_w and "property_quality" in work.columns: terms.append(F.coalesce(F.col("property_quality"), F.lit(0.0)) * F.lit(property_w))
    if host_w and "host_quality" in work.columns: terms.append(F.coalesce(F.col("host_quality"), F.lit(0.0)) * F.lit(host_w))

    # env components from osm_scores(): expects <env>_norm columns
    if env_weights:
        for env_col, raw_w in env_weights.items():
            wv = w_norm.get(f"env::{env_col}", raw_w) if normalize_all_weights else raw_w
            norm_col = env_col if env_col.endswith("_norm") else f"{env_col}_norm"
            if wv and norm_col in work.columns: terms.append(F.coalesce(F.col(norm_col), F.lit(0.0)) * F.lit(wv))

    # temperature component from cities_scores(): uses temp_avg_m01..temp_avg_m12
    if temp_w and temp_pref is not None and travel_month is not None:
        m = int(travel_month)
        if 1 <= m <= 12:
            temp_col = f"temp_avg_m{m:02d}"
            if temp_col in work.columns:
                work = work.withColumn("temp_score_raw", F.greatest(F.lit(0.0), F.lit(1.0) - (F.abs(F.col(temp_col) - F.lit(float(temp_pref))) / F.lit(25.0))))
                terms.append(F.col("temp_score_raw") * F.lit(temp_w))

    # budget component: prefer the score already computed in cities_scores() if present
    if budget_w and budget_pref is not None:
        rank_map = F.create_map(F.lit("Budget"), F.lit(1), F.lit("Mid-range"), F.lit(2), F.lit("Luxury"), F.lit(3))
        work = work.withColumn("budget_rank_i", rank_map.getItem(F.col("budget_level"))).withColumn("budget_pref_i", rank_map.getItem(F.lit(budget_pref)))
        work = work.withColumn("budget_score_raw_user", F.when(F.col("budget_rank_i").isNull() | F.col("budget_pref_i").isNull(), F.lit(0.0)).otherwise(F.greatest(F.lit(0.0), F.lit(1.0) - (F.abs(F.col("budget_rank_i") - F.col("budget_pref_i")) / F.lit(2.0)))))
        terms.append(F.col("budget_score_raw_user") * F.lit(budget_w))

    work = work.withColumn(score_col, F.lit(0.0)) if not terms else work.withColumn(score_col, sum(terms))
    return work.orderBy(F.col(score_col).desc()).limit(int(k))
