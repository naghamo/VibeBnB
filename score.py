# scoring_job.py

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.storagelevel import StorageLevel
import time

from config import ENV_COLS


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _has_cols(df: DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _as_double(c):
    return F.col(c).cast("double")


def _safe_div(num, den, default=0.0):
    return F.when(
        den.isNull() | (den == 0) | num.isNull(),
        F.lit(float(default))
    ).otherwise(num / den)


# ------------------------------------------------------------
# Scores
# ------------------------------------------------------------
def airbnb_scores(df: DataFrame) -> DataFrame:
    w_cc = Window.partitionBy("addr_cc")
    out = df

    # ---------- PRICE ----------
    out = out.withColumn("PPG", _safe_div(_as_double("price_per_night"), _as_double("guests")))
    out = out.withColumn("PPG_min", F.min("PPG").over(w_cc)) \
             .withColumn("PPG_max", F.max("PPG").over(w_cc))
    out = out.withColumn(
        "price_score",
        F.when(
            (F.col("PPG_max") == F.col("PPG_min")) | F.col("PPG").isNull(),
            F.lit(0.0)
        ).otherwise(
            1.0 - (F.col("PPG") - F.col("PPG_min")) / (F.col("PPG_max") - F.col("PPG_min"))
        )
    )

    # ---------- PROPERTY QUALITY ----------
    cat_cols = [
        "rating_accuracy", "rating_cleanliness", "rating_checkin",
        "rating_communication", "rating_location", "rating_value"
    ]
    cat_cols = _has_cols(out, cat_cols)

    out = out.withColumn(
        "category_rating_avg",
        sum(_as_double(c) for c in cat_cols) / F.lit(6.0)
    )
    out = out.withColumn("rating01", _safe_div(_as_double("ratings"), F.lit(5.0)))
    out = out.withColumn("category_rating01", _safe_div(_as_double("category_rating_avg"), F.lit(5.0)))
    out = out.withColumn(
        "rating_score",
        0.6 * F.col("rating01") + 0.4 * F.col("category_rating01")
    )

    out = out.withColumn(
        "reviews_log",
        F.log(F.coalesce(_as_double("property_number_of_reviews"), F.lit(0.0)) + 1.0)
    )
    out = out.withColumn("reviews_log_max", F.max("reviews_log").over(w_cc))
    out = out.withColumn(
        "reviews_weight",
        _safe_div(F.col("reviews_log"), F.col("reviews_log_max"))
    )

    out = out.withColumn(
        "property_quality",
        (
            F.col("rating_score") * F.col("reviews_weight")
            + 0.5 * (1.0 - F.col("reviews_weight"))
        ) * 0.8
        + F.coalesce(_as_double("is_guest_favorite"), F.lit(0.0)) * 0.2
    )

    # ---------- HOST QUALITY ----------
    out = out.withColumn("h_i", _safe_div(_as_double("host_rating"), F.lit(5.0)))
    out = out.withColumn("e_i", F.log(F.coalesce(_as_double("host_number_of_reviews"), F.lit(0.0)) + 1.0))
    out = out.withColumn("r_i", _safe_div(_as_double("host_response_rate"), F.lit(100.0)))
    out = out.withColumn("T_i", _as_double("hosts_year"))

    out = out.withColumn("e_max", F.max("e_i").over(w_cc))
    out = out.withColumn("T_max", F.max("T_i").over(w_cc))

    out = out.withColumn("e_norm", _safe_div(F.col("e_i"), F.col("e_max")))
    out = out.withColumn(
        "h_conf",
        F.col("e_norm") * F.col("h_i") + (1.0 - F.col("e_norm")) * 0.5
    )
    out = out.withColumn("T_norm", _safe_div(F.col("T_i"), F.col("T_max")))
    out = out.withColumn("tenure_effect", F.col("T_norm") * (F.col("h_i") - 0.5))

    out = out.withColumn(
        "host_quality",
        0.6 * F.col("h_conf")
        + 0.1 * F.col("r_i")
        + 0.1 * F.coalesce(_as_double("is_supperhost"), F.lit(0.0))
        + 0.2 * F.col("tenure_effect")
    )

    return out


def osm_scores(df: DataFrame, env_cols: list[str]) -> DataFrame:
    w_cc = Window.partitionBy("addr_cc")
    out = df

    for c in _has_cols(out, env_cols):
        out = out.withColumn(f"{c}_max", F.max(_as_double(c)).over(w_cc))
        out = out.withColumn(
            f"{c}_norm",
            _safe_div(_as_double(c), F.col(f"{c}_max"))
        )

    return out


def cities_scores(df: DataFrame) -> DataFrame:
    out = df

    if "avg_temp_monthly" in out.columns:
        out = out.withColumn(
            "temp_map",
            F.from_json(
                "avg_temp_monthly",
                "map<string, struct<avg:double, max:double, min:double>>"
            )
        )
        for m in range(1, 13):
            out = out.withColumn(
                f"temp_avg_m{m:02d}",
                F.col("temp_map").getItem(str(m)).getField("avg")
            )

    rank_map = F.create_map(
        F.lit("Budget"), F.lit(1),
        F.lit("Mid-range"), F.lit(2),
        F.lit("Luxury"), F.lit(3),
    )
    out = out.withColumn("budget_rank", rank_map.getItem("budget_level"))
    return out


def scoring_all(df: DataFrame) -> DataFrame:
    t0 = time.perf_counter()
    print("[SCORING] Airbnb scores...")
    df = airbnb_scores(df)
    print(f"[SCORING] Airbnb scores done in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    print("[SCORING] OSM normalization...")
    df = osm_scores(df, ENV_COLS)
    print(f"[SCORING] OSM scores done in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    print("[SCORING] City-level scores...")
    df = cities_scores(df)
    print(f"[SCORING] City scores done in {time.perf_counter() - t0:.2f}s")

    return df


# Main job
if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()

    t_job = time.perf_counter()
    print("[SCORING] Job started")

    in_path = "dbfs:/vibebnb/data/europe_countries_joined"
    out_path = "dbfs:/vibebnb/data/europe_countries_scored.parquet"

    t0 = time.perf_counter()
    df = spark.read.parquet(in_path).persist(StorageLevel.MEMORY_AND_DISK)
    count = df.count()
    print(f"[SCORING] Loaded {count:,} rows in {time.perf_counter() - t0:.2f}s")

    df_scored = scoring_all(df)

    t0 = time.perf_counter()
    (
        df_scored
        .write
        .mode("overwrite")
        .partitionBy("addr_cc")
        .parquet(out_path)
    )
    print(f"[SCORING] Saved output in {time.perf_counter() - t0:.2f}s")

    df.unpersist()
    print(f"[SCORING] Job finished in {time.perf_counter() - t_job:.2f}s")
