# embedding_fit.py
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    SQLTransformer, RegexTokenizer, StopWordsRemover,
    HashingTF, IDF, VectorAssembler,
    Normalizer, BucketedRandomProjectionLSH
)
from config import *
import time


# -----------------------------
# Helpers
# -----------------------------
def infer_env_embedding_cols(df: DataFrame) -> list[str]:
    """
    Infer which environment columns to use in the embedding:
    - Prefer normalized env_*_norm columns when present.
    - Otherwise fall back to raw env_* count columns.
    """
    env_norm = [c for c in df.columns if c.startswith("env_") and c.endswith("_norm")]
    if env_norm:
        return sorted(env_norm)
    env_raw = [c for c in df.columns if c.startswith("env_") and not c.endswith("_max") and not c.endswith("_norm")]
    return sorted(env_raw)


def _present_cols(df: DataFrame, cols: list[str]) -> list[str]:
    """Return only columns that exist in the dataframe (defensive schema handling)."""
    return [c for c in cols if c in df.columns]


# -----------------------------
# Build embedding pipeline
# -----------------------------
def fit_embedding_pipeline(df: DataFrame, text_cols: list[str]) -> Pipeline:
    """
    Fit a global embedding pipeline that creates a single feature vector per listing:
      - Text: concatenate selected text fields -> tokenize -> TF-IDF
      - Numeric: scored features + environment features
      - Final: concatenate numeric/env vector with TF-IDF text vector
    """
    text_cols = _present_cols(df, text_cols)

    # Combine text columns into a single lowercase string (robust to nulls).
    if text_cols:
        text_sql = " || ' ' || ".join([f"coalesce({c}, '')" for c in text_cols])
    else:
        text_sql = "''"

    combine_text = SQLTransformer(
        statement=f"SELECT *, lower(trim({text_sql})) AS text_all FROM __THIS__"
    )

    # Basic tokenization + stopword removal + TF-IDF representation.
    tokenizer = RegexTokenizer(
        inputCol="text_all", outputCol="tokens", pattern="\\W+", minTokenLength=2
    )
    stop_rm = StopWordsRemover(inputCol="tokens", outputCol="tokens_clean")
    hash_tf = HashingTF(inputCol="tokens_clean", outputCol="tf", numFeatures=1 << 18)
    idf = IDF(inputCol="tf", outputCol="text_tfidf")

    # --- Numeric cols ---
    # SCORED_NUM_COLS comes from config and represents the core numeric scoring features.
    scored_cols = _present_cols(df, SCORED_NUM_COLS)

    # Environment columns (normalized if available, else raw counts).
    env_cols = infer_env_embedding_cols(df)

    # Assemble numeric/env features into a single vector.
    num_env_inputs = scored_cols + env_cols
    num_env_assembler = VectorAssembler(
        inputCols=num_env_inputs,
        outputCol="num_env_vec",
        handleInvalid="keep"
    )

    # Final feature vector: concatenate [numeric/env] with [TF-IDF text] into "features".
    final_assembler = VectorAssembler(
        inputCols=["num_env_vec", "text_tfidf"],
        outputCol="features",
        handleInvalid="keep"
    )

    # Pipeline stages in execution order.
    stages = [combine_text, tokenizer, stop_rm, hash_tf, idf, num_env_assembler, final_assembler]

    pipe = Pipeline(stages=stages)
    return pipe.fit(df)


def embed(df: DataFrame, emb_model) -> DataFrame:
    """
    Apply the fitted embedding model to produce L2-normalized feature vectors.
    Output includes only (property_id, addr_cc, features_norm) for downstream retrieval.
    """
    df_feat = emb_model.transform(df)

    # L2 normalization enables cosine similarity to be computed via Euclidean distance.
    normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=2.0)

    keep_cols = [c for c in ["property_id", "addr_cc"] if c in df_feat.columns]
    return normalizer.transform(df_feat).select(*keep_cols, "features_norm")


def build_lsh(df_emb: DataFrame) -> BucketedRandomProjectionLSH:
    """
    Fit a Bucketed Random Projection LSH index over normalized feature vectors.
    This supports approximate nearest-neighbor retrieval using Euclidean distance.
    """
    lsh = BucketedRandomProjectionLSH(
        inputCol="features_norm",
        outputCol="hashes",
        bucketLength=0.5,
        numHashTables=3
    )
    return lsh.fit(df_emb)


def build_global_models_and_save(
    scored_input_path: str,
    embedded_out_path: str,
    emb_model_path: str,
    lsh_model_path: str
):
    """
    End-to-end embedding job:
      1) Load scored dataset
      2) Fit global embedding pipeline (text + numeric/env)
      3) Transform to features_norm
      4) Fit global LSH model
      5) Save embedding pipeline, LSH model, and embedded dataset (partitioned by country)
    """
    spark = SparkSession.builder.getOrCreate()

    t_job = time.perf_counter()
    print("[EMB] Job started")

    # Load scored data (expected to include all features required by config columns).
    t0 = time.perf_counter()
    df = spark.read.parquet(scored_input_path).persist(StorageLevel.MEMORY_AND_DISK)
    n = df.count()
    print(f"[EMB] Loaded {n:,} rows from {scored_input_path} in {time.perf_counter()-t0:.2f}s")

    # Fit embedding pipeline (global fit so IDF weights are consistent across countries).
    print("[EMB] Fitting global embedding pipeline...")
    t0 = time.perf_counter()
    emb_model = fit_embedding_pipeline(df, TEXT_COLS_DEFAULT)
    print(f"[EMB] Pipeline fit done in {time.perf_counter()-t0:.2f}s")

    # Transform -> features_norm (ready for similarity computations).
    print("[EMB] Transforming to features_norm...")
    t0 = time.perf_counter()
    df_emb = embed(df, emb_model).persist(StorageLevel.MEMORY_AND_DISK)
    m = df_emb.count()
    print(f"[EMB] Transform produced {m:,} rows in {time.perf_counter()-t0:.2f}s")

    # Fit global LSH model over normalized embeddings.
    print("[LSH] Training global LSH model...")
    t0 = time.perf_counter()
    lsh_model = build_lsh(df_emb)
    print(f"[LSH] LSH fit done in {time.perf_counter()-t0:.2f}s")

    # Save models for reuse in offline/online retrieval steps.
    print(f"[SAVE] Embedding pipeline -> {emb_model_path}")
    emb_model.write().overwrite().save(emb_model_path)

    print(f"[SAVE] LSH model -> {lsh_model_path}")
    lsh_model.write().overwrite().save(lsh_model_path)

    # Save embedded dataset (partitioned by addr_cc for country-scoped loading).
    print(f"[SAVE] Embedded dataset -> {embedded_out_path} (partitionBy addr_cc)")
    (
        df_emb
        .write
        .mode("overwrite")
        .partitionBy("addr_cc")
        .parquet(embedded_out_path)
    )

    df.unpersist()
    df_emb.unpersist()

    print(f"[DONE] Finished in {time.perf_counter()-t_job:.2f}s")


# -----------------------------
# Run
# -----------------------------
# Default execution entrypoint for training and saving global embedding + LSH artifacts.
build_global_models_and_save(
    scored_input_path="dbfs:/vibebnb/data/europe_countries_scored_.parquet",
    embedded_out_path="dbfs:/vibebnb/data/europe_countries_embedded_",
    emb_model_path="dbfs:/vibebnb/models/embedding_pipeline_global_",
    lsh_model_path="dbfs:/vibebnb/models/lsh_global_"
)
