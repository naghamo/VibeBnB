import time
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel
from pyspark.ml.feature import BucketedRandomProjectionLSHModel
from retrieve_rank import retrieve, order
from config import *
import warnings

warnings.filterwarnings("ignore")

def infer_env_cols_from_columns(cols: list[str]) -> list[str]:
    """
    Infers environmental columns from a list of column names.

    :param cols: List of column names.
    :type cols: list[str]
    :returns: Sorted list of environmental column names.
    :rtype: list[str]
    """
    env_norm = [c for c in cols if c.startswith("env_") and c.endswith("_norm")]
    if env_norm:
        return sorted(env_norm)
    env_raw = [c for c in cols if c.startswith("env_") and (not c.endswith("_max")) and (not c.endswith("_norm"))]
    return sorted(env_raw)

save_cols = [
    "property_id", "addr_cc", "lat", "long", "listing_title", "room_type_text",
    "addr_name", "price_per_night", "ratings", 'l2_dist', 'cosine_similarity', "final_url"
]

# User preferences and configuration
temp_pref = 22
travel_month = 6
budget_pref = "Mid-range"

def get_configs(env_cols):
    return {
        "airbnb": {
            "price_w": 90,
            "property_w": 90,
            "host_w": 90,
            "env_weights": {col: 10 for col in env_cols},
            "temp_pref": temp_pref,
            "temp_w": 10,
            "travel_month": travel_month,
            "budget_pref": budget_pref,
            "budget_w": 10,
            "normalize_all_weights": True,
            "score_col": "final_score"
        },
        "cities": {
            "price_w": 10,
            "property_w": 10,
            "host_w": 10,
            "env_weights": {col: 10 for col in env_cols},
            "temp_pref": temp_pref,
            "temp_w": 90,
            "travel_month": travel_month,
            "budget_pref": budget_pref,
            "budget_w": 90,
            "normalize_all_weights": True,
            "score_col": "final_score"
        },
        "env": {
            "price_w": 10,
            "property_w": 10,
            "host_w": 10,
            "env_weights": {col: 90 for col in env_cols},
            "temp_pref": temp_pref,
            "temp_w": 10,
            "travel_month": travel_month,
            "budget_pref": budget_pref,
            "budget_w": 10,
            "normalize_all_weights": True,
            "score_col": "final_score"
        }
    }

def retrieve_for_id(id: int, dest_cc: str, MCQS_df):
    """
    Retrieves candidate properties for a given property ID and destination country code.
    Saves the top 5 candidates to a CSV file.

    :param id: Target property ID.
    :type id: int
    :param dest_cc: Destination country code.
    :type dest_cc: str
    :param MCQS_df: MCQS results DataFrame.
    :returns: Spark DataFrame of candidate properties.
    :rtype: DataFrame
    """
    cand_df = MCQS_df.filter(
        (MCQS_df.target_id == id) & (MCQS_df.cand_cc == dest_cc)
    ).orderBy(F.col("l2_dist").asc())
    pdf = cand_df.select(save_cols).limit(5).toPandas()
    pdf.to_csv(f"{id}_{dest_cc}.csv", index=False)
    return cand_df

def save_reordered(cand_df, id: int, dest_cc: str, configs):
    """
    Reorders candidate DataFrame using different configuration settings and saves results to CSV files.

    :param cand_df: Candidate properties DataFrame.
    :type cand_df: DataFrame
    :param id: Target property ID.
    :type id: int
    :param dest_cc: Destination country code.
    :type dest_cc: str
    :param configs: Configuration dictionary.
    :returns: None
    """
    for cfg_name, cfg in configs.items():
        print(cfg_name)
        cand_df_reordered = order(
            df=cand_df,
            k=5,
            price_w=cfg["price_w"],
            property_w=cfg["property_w"],
            host_w=cfg["host_w"],
            env_weights=cfg["env_weights"],
            temp_pref=cfg["temp_pref"],
            temp_w=cfg["temp_w"],
            travel_month=cfg["travel_month"],
            budget_pref=cfg["budget_pref"],
            budget_w=cfg["budget_w"],
            normalize_all_weights=cfg["normalize_all_weights"],
            score_col=cfg["score_col"]
        )
        pdf = cand_df_reordered.select(['final_score'] + save_cols).toPandas()
        pdf.to_csv(f"{cfg_name}_{id}_{dest_cc}.csv", index=False)

# ---- Run ----
def run_pipeline(df_emb, df_all, MCQS_df):
    ENV_COLS = infer_env_cols_from_columns(df_all.columns)
    configs = get_configs(ENV_COLS)
    queries = {47067457: "FR"} # add any other queries if needed
    for query_id, dest_cc in queries.items():
        cand_df = retrieve_for_id(query_id, dest_cc, MCQS_df)
        save_reordered(cand_df, query_id, dest_cc, configs)
