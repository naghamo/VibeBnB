from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.storagelevel import StorageLevel
from pyspark.ml.feature import BucketedRandomProjectionLSHModel
from config import *

# ----------------------------
# Config
# ----------------------------
STAGING_DIR = "dbfs:/vibebnb/offline_static_build"
NEIGHBORS_DBFS_DIR = f"{STAGING_DIR}/neighbors"

SAMPLE_SIZE = 500
TOP_N = 50
TARGET_CCS = ["FR", "IT", "ES", "GB", "DE"]
DIST_THRESHOLD = 10.0
#############################################!!change this to your path!!###########################################
REPO_STATIC_DIR = "/Workspace/Users/nagham.omar@campus.technion.ac.il/VibeBnB/static"

SAMPLE_JSON_PATH = os.path.join(REPO_STATIC_DIR, "listings_sample_.json")
NEIGHBORS_JSON_DIR = os.path.join(REPO_STATIC_DIR, "neighbors_json_")

UI_COLS = [
    "property_id", "addr_cc", "listing_title", "room_type_text",
    "addr_name", "price_per_night", "ratings"
]

DISPLAY_COLS = ["listing_title", "room_type_text", "addr_name", "price_per_night", "ratings", "final_url"]

CORE_SCORE_COLS = ["price_score", "property_quality", "host_quality"]
ENV_NORM_COLS = [
    "env_culture_norm","env_family_norm","env_food_norm","env_health_norm",
    "env_leisure_norm","env_nature_norm","env_nightlife_norm","env_services_norm",
    "env_shopping_norm","env_sightseeing_norm","env_supplies_norm","env_transport_norm"
]
TEMP_MONTHLY_COLS = [f"temp_avg_m{m:02d}" for m in range(1, 13)]
CITY_COMPONENT_COLS = ["budget_rank"] + TEMP_MONTHLY_COLS

RANKING_COLS = CORE_SCORE_COLS + ENV_NORM_COLS + CITY_COMPONENT_COLS


# ----------------------------
# Utils
# ----------------------------
def _ensure_dirs() -> None:
    os.makedirs(REPO_STATIC_DIR, exist_ok=True)
    os.makedirs(NEIGHBORS_JSON_DIR, exist_ok=True)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def topn_from_pairs(pairs_df: DataFrame, top_n: int) -> DataFrame:
    flat = (
        pairs_df.select(
            F.col("datasetA.property_id").alias("src_property_id"),
            F.col("datasetB.property_id").alias("neighbor_property_id"),
            F.col("datasetB.addr_cc").alias("target_cc"),
            F.col("distCol").alias("l2_dist"),
        )
        .filter(F.col("datasetA.property_id") != F.col("datasetB.property_id"))
    )
    w = Window.partitionBy("src_property_id", "target_cc").orderBy(F.col("l2_dist").asc())
    return (
        flat.withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") <= F.lit(int(top_n)))
        .drop("rn")
    )


# ----------------------------
# Steps
# ----------------------------
def sample_listings(spark, n: int = SAMPLE_SIZE) -> List[Dict[str, Any]]:
    _ensure_dirs()
    df = (
        spark.read.parquet(SCORED_DATA_PATH)
        .select(*UI_COLS)
        .where(F.col("property_id").isNotNull())
        .where(F.col("addr_cc").isNotNull())
        .orderBy(F.rand())
        .limit(int(n))
        .withColumn("property_id", F.col("property_id").cast("string"))
    )
    rows = [r.asDict(recursive=True) for r in df.collect()]
    _write_json(SAMPLE_JSON_PATH, rows)
    print(f" Sample saved: {SAMPLE_JSON_PATH} | rows={len(rows)}")
    return rows


def build_neighbors_parquet(spark, dbutils, sample_ids: List[str]) -> None:
    lsh_model = BucketedRandomProjectionLSHModel.load(LSH_MODEL_PATH)

    df_emb = (
        spark.read.parquet(EMBEDDED_PATH)
        .select(F.col("property_id").cast("string").alias("property_id"),
                "addr_cc", "features_norm")
        .dropDuplicates(["property_id"])
        .where(F.col("addr_cc").isin(TARGET_CCS))
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    _ = df_emb.count()

    df_full_all = spark.read.parquet(SCORED_DATA_PATH)
    existing = set(df_full_all.columns)
    needed = ["property_id", "addr_cc"] + DISPLAY_COLS + RANKING_COLS
    select_cols = [c for c in needed if c in existing]

    df_full = (
        df_full_all.select(*select_cols)
        .withColumn("property_id", F.col("property_id").cast("string"))
        .dropDuplicates(["property_id"])
        .where(F.col("addr_cc").isin(TARGET_CCS))
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    _ = df_full.count()

    df_sample_ids = spark.createDataFrame([(x,) for x in sample_ids], ["property_id"])
    df_sample = (
        df_sample_ids.join(df_emb, on="property_id", how="inner")
        .select("property_id", "addr_cc", "features_norm")
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    if df_sample.count() == 0:
        raise RuntimeError("No sampled IDs matched EMBEDDED_PATH.")

    dbutils.fs.rm(NEIGHBORS_DBFS_DIR, True)

    for cc in TARGET_CCS:
        print(f"\n[BUILD] target_cc={cc}")
        df_cc = df_emb.where(F.col("addr_cc") == cc).select("property_id", "addr_cc", "features_norm").persist()
        if df_cc.count() == 0:
            df_cc.unpersist()
            print("  skip")
            continue

        t0 = time.perf_counter()
        pairs = lsh_model.approxSimilarityJoin(df_sample, df_cc, float(DIST_THRESHOLD), distCol="distCol")
        neigh = topn_from_pairs(pairs, TOP_N)

        neigh = (
            neigh.join(df_full, neigh.neighbor_property_id == df_full.property_id, "left")
                .drop(df_full.property_id)
        )

        out_path = f"{NEIGHBORS_DBFS_DIR}/target_cc={cc}"
        neigh.write.mode("overwrite").parquet(out_path)
        print(f"  wrote {out_path} | rows={neigh.count()} | {(time.perf_counter()-t0):.2f}s")

        df_cc.unpersist()

    df_sample.unpersist()
    df_emb.unpersist()
    df_full.unpersist()
    print(f"\n Neighbors parquet saved: {NEIGHBORS_DBFS_DIR}")


def export_neighbors_json(spark, sample_ids: List[str]) -> None:
    _ensure_dirs()

    for cc in TARGET_CCS:
        src_path = f"{NEIGHBORS_DBFS_DIR}/target_cc={cc}"
        out_path = os.path.join(NEIGHBORS_JSON_DIR, f"target_cc={cc}.json")

        df = spark.read.parquet(src_path).withColumn("src_property_id", F.col("src_property_id").cast("string"))
        df = df.filter(F.col("src_property_id").isin(sample_ids))

        # safety (should already be top-50)
        if "l2_dist" in df.columns:
            w = Window.partitionBy("src_property_id").orderBy(F.col("l2_dist").asc_nulls_last())
            df = (
                df.withColumn("rn", F.row_number().over(w))
                .filter(F.col("rn") <= F.lit(int(TOP_N)))
                .drop("rn")
            )

        cols = [c for c in df.columns if c != "src_property_id"]
        grouped = df.groupBy("src_property_id").agg(F.collect_list(F.struct(*cols)).alias("results"))

        mapping: Dict[str, List[Dict[str, Any]]] = {}
        for r in grouped.collect():
            sid = r["src_property_id"]
            mapping[sid] = [x.asDict(recursive=True) for x in r["results"]]

        _write_json(out_path, mapping)
        print(f"JSON saved: {out_path} | keys={len(mapping)}")


def main(spark, dbutils) -> None:
    _ensure_dirs()
    if not os.path.exists(SAMPLE_JSON_PATH):
        sample_listings(spark, SAMPLE_SIZE)

    sample_rows = _read_json(SAMPLE_JSON_PATH)
    sample_ids = sorted({str(r["property_id"]) for r in sample_rows if r.get("property_id") is not None})

    build_neighbors_parquet(spark, dbutils, sample_ids)
    export_neighbors_json(spark, sample_ids)
    print("\nDone. Static assets are ready.")



main(spark, dbutils)
