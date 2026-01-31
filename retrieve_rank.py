# retrieve.py / rank.py
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors, DenseVector
import numpy as np
from pyspark.ml.functions import vector_to_array


def retrieve(target_id, country, df, lsh_model, n=50):
    """
    Approximate nearest-neighbor retrieval (ANN) within a single target country.

    Inputs:
    - target_id: property_id of the reference listing
    - country: destination country code (addr_cc) to search within
    - df: embedded dataframe containing (property_id, addr_cc, features_norm)
    - lsh_model: trained BucketedRandomProjectionLSH model on features_norm
    - n: number of neighbors to retrieve

    Output:
    - DataFrame of top-n nearest neighbors within the destination country, with l2_dist column
      (or None if the target_id is missing from df).
    """
    # Fetch the query embedding vector for the target listing.
    q = df.filter(F.col("property_id") == target_id).select("features_norm").limit(1).collect()
    if not q:
        return None

    q_vec = q[0]["features_norm"]

    # Candidate pool restricted to one destination country (keeps retrieval localized).
    cand = df.filter(F.col("addr_cc") == country)

    # LSH returns approximate neighbors using Euclidean distance over normalized vectors.
    return lsh_model.approxNearestNeighbors(cand, q_vec, n, distCol="l2_dist")


def _normalize_weights(w: dict):
    """
    Normalize positive weights to sum to 1.

    Rules:
    - Ignore None and non-positive values.
    - If the sum is <= 0, return empty dict (meaning "no active weights").
    """
    if not w:
        return {}
    s = sum(v for v in w.values() if v is not None and v > 0)
    if s <= 0:
        return {}
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
    """
    Preference-aware ranking over a pre-filtered candidate DataFrame.

    This function assumes candidate rows already include precomputed score components:
    - price_score, property_quality, host_quality (from Airbnb scoring)
    - env_*_norm (from OSM normalization)
    - temp_avg_m01..temp_avg_m12 and budget_level (from city-level join/parsing)

    Behavior:
    - Builds a weighted linear combination of available components.
    - Optionally normalizes *all* provided weights together (including env weights).
    - Returns top-k rows ordered by the computed score_col (descending).
    """
    work = df

    # Collect all weights into one dictionary for normalization.
    env_weights = env_weights or {}
    all_w = {
        "price": price_w,
        "property": property_w,
        "host": host_w,
        "temp": temp_w,
        "budget": budget_w,
        **{f"env::{k}": v for k, v in env_weights.items()}
    }

    # Normalize weights to sum to 1.
    w_norm = _normalize_weights(all_w) if normalize_all_weights else all_w

    # Extract normalized (or raw) top-level weights.
    price_w = w_norm.get("price", 0.0)
    property_w = w_norm.get("property", 0.0)
    host_w = w_norm.get("host", 0.0)
    temp_w = w_norm.get("temp", 0.0)
    budget_w = w_norm.get("budget", 0.0)

    terms = []

    # --- Base components (Airbnb scores) ---
    if price_w and "price_score" in work.columns:
        terms.append(F.coalesce(F.col("price_score"), F.lit(0.0)) * F.lit(price_w))
    if property_w and "property_quality" in work.columns:
        terms.append(F.coalesce(F.col("property_quality"), F.lit(0.0)) * F.lit(property_w))
    if host_w and "host_quality" in work.columns:
        terms.append(F.coalesce(F.col("host_quality"), F.lit(0.0)) * F.lit(host_w))

    # --- Environment components (OSM) ---
    # Expects normalized env columns: env_xxx_norm. If user passed env_xxx, we map to env_xxx_norm.
    if env_weights:
        for env_col, raw_w in env_weights.items():
            # If normalize_all_weights=True, use the normalized weight; otherwise use the raw weight.
            wv = w_norm.get(f"env::{env_col}", raw_w) if normalize_all_weights else raw_w

            # Accept both "env_food" and "env_food_norm" as inputs.
            norm_col = env_col if env_col.endswith("_norm") else f"{env_col}_norm"

            if wv and norm_col in work.columns:
                terms.append(F.coalesce(F.col(norm_col), F.lit(0.0)) * F.lit(wv))

    # --- Temperature component (City-level) ---
    # Uses city monthly average temperature temp_avg_m01..temp_avg_m12, with tolerance 25C.
    if temp_w and (temp_pref is not None) and (travel_month is not None):
        m = int(travel_month)
        if 1 <= m <= 12:
            temp_col = f"temp_avg_m{m:02d}"
            if temp_col in work.columns:
                work = work.withColumn(
                    "temp_score_raw",
                    F.when(
                        F.col(temp_col).isNull(),
                        F.lit(0.0)
                    ).otherwise(
                        F.greatest(
                            F.lit(0.0),
                            F.lit(1.0) - (F.abs(F.col(temp_col) - F.lit(float(temp_pref))) / F.lit(25.0))
                        )
                    )
                )
                terms.append(F.col("temp_score_raw") * F.lit(temp_w))

    # --- Budget component (City-level) ---
    # Computes a user-alignment score from budget_level, mapped to ranks 1..3.
    if budget_w and (budget_pref is not None):
        rank_map = F.create_map(
            F.lit("Budget"), F.lit(1),
            F.lit("Mid-range"), F.lit(2),
            F.lit("Luxury"), F.lit(3),
        )
        work = work.withColumn("budget_rank_i", rank_map.getItem(F.col("budget_level")))
        work = work.withColumn("budget_pref_i", rank_map.getItem(F.lit(budget_pref)))
        work = work.withColumn(
            "budget_score_raw_user",
            F.when(
                F.col("budget_rank_i").isNull() | F.col("budget_pref_i").isNull(),
                F.lit(0.0)
            ).otherwise(
                F.greatest(
                    F.lit(0.0),
                    F.lit(1.0) - (F.abs(F.col("budget_rank_i") - F.col("budget_pref_i")) / F.lit(2.0))
                )
            )
        )
        terms.append(F.col("budget_score_raw_user") * F.lit(budget_w))

    # If no terms exist, score is 0 for all rows; otherwise sum weighted terms.
    work = work.withColumn(score_col, F.lit(0.0)) if not terms else work.withColumn(score_col, sum(terms))

    # Return top-k by computed score.
    return work.orderBy(F.col(score_col).desc()).limit(int(k))
