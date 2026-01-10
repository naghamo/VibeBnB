# embedding_job.py
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    SQLTransformer, RegexTokenizer, StopWordsRemover,
    HashingTF, IDF, VectorAssembler, StandardScaler,
    Normalizer, BucketedRandomProjectionLSH
)
from config import *
import time




def infer_env_embedding_cols(df: DataFrame) -> list[str]:
    env_norm = [c for c in df.columns if c.startswith("env_") and c.endswith("_norm")]
    if env_norm:
        return sorted(env_norm)
    env_raw = [c for c in df.columns if c.startswith("env_") and not c.endswith("_max") and not c.endswith("_norm")]
    return sorted(env_raw)


# -----------------------------
# Text columns 
# -----------------------------

def _present_cols(df: DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]



# Build embedding pipeline
def fit_embedding_pipeline(df: DataFrame, text_cols: list[str]) -> Pipeline:
    text_cols = _present_cols(df, text_cols)
    # Combine text columns 
    if text_cols:
        text_sql = " || ' ' || ".join([f"coalesce({c}, '')" for c in text_cols])
    else:
        text_sql = "''"

    combine_text = SQLTransformer(
        statement=f"SELECT *, lower(trim({text_sql})) AS text_all FROM __THIS__"
    )

    tokenizer = RegexTokenizer(
        inputCol="text_all", outputCol="tokens", pattern="\\W+", minTokenLength=2
    )
    stop_rm = StopWordsRemover(inputCol="tokens", outputCol="tokens_clean")
    hash_tf = HashingTF(inputCol="tokens_clean", outputCol="tf", numFeatures=1 << 18)
    idf = IDF(inputCol="tf", outputCol="text_tfidf")

    # Numeric cols AFTER scoring (+ env embedding cols inferred from DF)
    num_cols = _present_cols(df, SCORED_NUM_COLS)
    env_cols = infer_env_embedding_cols(df)

    num_env_assembler = VectorAssembler(
        inputCols=num_cols + env_cols,
        outputCol="num_env_vec",
        handleInvalid="keep"
    )

    scaler = StandardScaler(
        inputCol="num_env_vec",
        outputCol="num_env_scaled",
        withStd=True,
        withMean=False
    )

    final_assembler = VectorAssembler(
        inputCols=["num_env_scaled", "text_tfidf"],
        outputCol="features",
        handleInvalid="keep"
    )

    # Normalizer is applied after pipeline transform
    pipe = Pipeline(stages=[
        combine_text, tokenizer, stop_rm, hash_tf, idf,
        num_env_assembler, scaler, final_assembler
    ])

    return pipe.fit(df)


def embed(df: DataFrame, emb_model) -> DataFrame:
    df_feat = emb_model.transform(df)

    # L2 normalization for cosine similarity via Euclidean distance
    normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=2.0)

    # Keep addr_cc for partitioning + any debugging keys you want
    keep_cols = [c for c in ["property_id", "addr_cc"] if c in df_feat.columns]
    return normalizer.transform(df_feat).select(*keep_cols, "features_norm")


def build_lsh(df_emb: DataFrame) -> BucketedRandomProjectionLSH:
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
    spark = SparkSession.builder.getOrCreate()

    t_job = time.perf_counter()
    print("[EMB] Job started")

    #Load scored data 
    t0 = time.perf_counter()
    df = spark.read.parquet(scored_input_path).persist(StorageLevel.MEMORY_AND_DISK)
    n = df.count()  # materialize cache + sanity
    print(f"[EMB] Loaded {n:,} rows from {scored_input_path} in {time.perf_counter()-t0:.2f}s")

    #Fit embedding pipeline on all rows
    print("[EMB] Fitting global embedding pipeline...")
    t0 = time.perf_counter()
    emb_model = fit_embedding_pipeline(df, TEXT_COLS_DEFAULT)
    print(f"[EMB] Pipeline fit done in {time.perf_counter()-t0:.2f}s")

    #Transform -> features_norm
    print("[EMB] Transforming to features_norm...")
    t0 = time.perf_counter()
    df_emb = embed(df, emb_model).persist(StorageLevel.MEMORY_AND_DISK)
    m = df_emb.count()
    print(f"[EMB] Transform produced {m:,} rows in {time.perf_counter()-t0:.2f}s")

    #Fit ONE global LSH model
    print("[LSH] Training global LSH model...")
    t0 = time.perf_counter()
    lsh_model = build_lsh(df_emb)
    print(f"[LSH] LSH fit done in {time.perf_counter()-t0:.2f}s")

    # Save models
    print(f"[SAVE] Embedding pipeline -> {emb_model_path}")
    emb_model.write().overwrite().save(emb_model_path)

    print(f"[SAVE] LSH model -> {lsh_model_path}")
    lsh_model.write().overwrite().save(lsh_model_path)

    #Save embedded dataset
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
build_global_models_and_save(
    scored_input_path="dbfs:/vibebnb/data/europe_countries_scored.parquet",
    embedded_out_path="dbfs:/vibebnb/data/europe_countries_embedded",
    emb_model_path="dbfs:/vibebnb/models/embedding_pipeline_global",
    lsh_model_path="dbfs:/vibebnb/models/lsh_global"
)
