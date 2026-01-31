from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel
from pyspark.ml.feature import BucketedRandomProjectionLSHModel
from retrieve_rank import retrieve, order
from config import *
import matplotlib.pyplot as plt
import time
import random
import pandas as pd
import numpy as np
import warnings


warnings.filterwarnings("ignore")

N_PROPS_MCQS = 10
EUROPE_CC = set(continents['europe'])

def pick_diverse_props(df_all, df_emb, lsh_model, n_props, exclude_countries=[]):
    """
    Picks a series of properties from the dataset, trying to maximize the diversity of the resulting set by euclidean distance.

    :param df_all: A dataframe containing all properties and their data.
    :param df_emb: A dataframe containing the embeddings of all properties.
    :param lsh_model: A BucketedRandomProjectionLS model.
    :param n_props: The number of properties to pick.
    :param exclude_coutries: A list of countries to exclude from the search (that might be used as destinations later on).
    :return: A list of property ids.
    """
    # find all european countries in data
    countries_europe = (
    df_all.select("addr_cc")
          .where(F.col("addr_cc").isNotNull())
          .select(F.upper(F.col("addr_cc")).alias("addr_cc"))
          .distinct()
          .where(F.col("addr_cc").isin(list(EUROPE_CC)))
          .orderBy("addr_cc")
          .toPandas()["addr_cc"]
          .tolist()
    )

    countries_europe = list(set(countries_europe) - set(exclude_countries))

    # country value counts df will be used to search for suitable queries from all data
    country_value_counts = (
    df_all.select("addr_cc")
          .groupBy("addr_cc")
          .count()
          .toPandas()
    )

    print("[QSM] Initializing variables...")
    cc_pool = countries_europe.copy()
    queries = pd.DataFrame(columns=['property_id','country_code'])
    seed_country = random.choice(cc_pool)
    cc_pool.remove(seed_country)

    print("[QSM] Picking the first seed property...")
    t_0 = time.perf_counter()
    # pick a random property from that country
    seed_prop_id = (
        df_all.select(['property_id', 'addr_cc'])
            .where(F.col("addr_cc") == seed_country)
            .orderBy(F.rand())
            .limit(1)
            .toPandas()["property_id"].values[0]
    )
    t_1 = time.perf_counter()
    print(f"[QSM] Found initial seed property in {(t_1 - t_0):.2f}s.")
    print(f"[QSM] Starting query suite building process from property {seed_prop_id} in {seed_country}!")

    queries = queries.append({'property_id': seed_prop_id, 'country_code': seed_country}, ignore_index=True)

    for i in range(n_props-1):
        # pick the next random country
        dest_cc = random.choice(cc_pool)
        cc_pool.remove(dest_cc)

        print("\n---------------------------")
        print(f"Let's travel from {seed_country} to {dest_cc}!")
        print(f"Calculating least similar property to {len(queries)} previous query properties...")

        t_0 = time.perf_counter()

        # try to find a property in that country that is farthest from the seed property
        n_listings = country_value_counts.loc[country_value_counts["addr_cc"] == dest_cc, "count"].values[0]
        props_distances = None

        for i, seed_prop_id in enumerate(queries["property_id"]):
            print(f"- approaching from property n. {i+1}, {seed_prop_id}")
            i_props_distances = (
                retrieve(int(seed_prop_id), dest_cc, df_emb, lsh_model, int(n_listings))
                .select(['property_id', 'l2_dist'])
                .withColumnRenamed('l2_dist', f'l2_dist_{i}')
            )
            if props_distances:
                props_distances = props_distances.join(i_props_distances, on="property_id", how="inner")
            else:
                props_distances = i_props_distances

        cols_to_sum = [f'l2_dist_{i}' for i in range(len(queries))]
        sum_expr = '+'.join(cols_to_sum)
        props_distances = props_distances.withColumn("total_dist", F.expr(sum_expr))
        furthest_prop = props_distances.orderBy(F.col("total_dist").desc()).limit(1)
        furthest_prop_id = furthest_prop.toPandas()["property_id"].values[0]
        total_dist = furthest_prop.toPandas()["total_dist"].values[0]

        t_1 = time.perf_counter()

        print(f"Retrieved property {furthest_prop_id} with total distance {total_dist:.2f}, from country {dest_cc} in {(t_1 - t_0):.2f}s.")

        queries = queries.append({'property_id': furthest_prop_id, 'country_code': dest_cc}, ignore_index=True)

        # propagate
        seed_country = dest_cc
        seed_prop_id = furthest_prop_id

    return queries, cc_pool


def pick_smallest_and_largest_ccs(df_all, n_from_each=3):
    """
    Pick the smallest and largest country codes by frequency, and return them as lists.
    Only consider countries with at least 50 observations.
    :param df_all: the entire airbnb dataset
    :param n_from_each: the number of country codes to return for each of the two groups
    :return: a tuple of two lists, the first being the smallest country codes, the second being the largest
    """
    country_value_counts = (
        df_all.select(['addr_cc'])
            .groupBy('addr_cc')
            .count()
            .where(F.col('count') >= 50)
            .orderBy(F.col('count').asc())
    )
    smallest_ccs = country_value_counts.limit(n_from_each).toPandas()['addr_cc'].values
    largest_ccs = country_value_counts.orderBy(F.col('count').desc()).limit(n_from_each).toPandas()['addr_cc'].values
    return list(smallest_ccs), list(largest_ccs)


def build_OCQS(df_all, df_emb, lsh_model, n_queries=24):
    """
    Builds a query suite, where each query is a property and a destination country.
    The queries are picked so that they will be diverse - they are far from each other in the embedding space.
    Here, each query recieves one random destination country.

    :param df_all: the entire dataset
    :param df_emb: the embedding dataframe
    :param lsh_model: the LSH model
    :param n_queries: the number of queries to build
    :return: DataFrame of queries with property_id, country_code, and dest_country columns
    """
    queries, cc_pool = pick_diverse_props(df_all, df_emb, lsh_model, n_props=n_queries)
    dest_countries = random.sample(cc_pool, k=n_queries)
    queries['dest_country'] = dest_countries

    print("\nFinished.")

    return queries

def build_MCQS(df_all, df_emb, lsh_model, n_props=10, countries=3):
    """
    Builds a query suite, where each query is a property and a destination country.
    The queries are picked so that they will be diverse - they are far from each other in the embedding space.
    countries is a polymorphic argument, and can either be a specified list of countries, or an integer, in which case
    the function will pick a random sample of countries from the pool of countries in the dataset.

    :param df_all: the entire dataset
    :param df_emb: the embedding dataframe
    :param lsh_model: the LSH model
    :param n_props: the number of diverse properties to pick
    :param countries: either a specified list of countries, or an integer specifying the number of countries to pick
    :return: tuple of (queries DataFrame, targets_data DataFrame)
    """

    dest_countries = []
    if isinstance(countries, int):
        dest_countries = random.sample(cc_pool, k=countries)
    elif isinstance(countries, list):
        dest_countries = countries

    queries, cc_pool = pick_diverse_props(df_all, df_emb, lsh_model, n_props=n_props, exclude_countries=dest_countries)
    targets_data = df_all.join(spark.createDataFrame(queries), on='property_id', how='inner')
    
    queries = queries.merge(pd.DataFrame(dest_countries, columns=["dest_country"]), how="cross")

    print("\nFinished.")

    return queries, targets_data

def retrieve_suite_results(df_all, df_emb, lsh_model, query_suite):
    """
    Retrieve the top 50 candidates for each query in the query suite and return a DataFrame with results.

    :param df_all: DataFrame containing all property data
    :param df_emb: DataFrame containing property embeddings
    :param lsh_model: Trained LSH model for similarity search
    :param query_suite: DataFrame or pandas DataFrame with queries (property_id, country_code, dest_country)
    :return: Spark DataFrame with retrieval results, including candidate properties and similarity metrics
    """

    # Retrieve top 50 for each query, and save it to a dataframe
    if not type(query_suite) == pd.DataFrame:
        query_suite_pd = query_suite.toPandas()
    else:
        query_suite_pd = query_suite
        
    suite_results = None
    n_candidates = 50
    print(f"[QSM] Starting retrieval for {len(query_suite_pd)} queries...")

    for i, row in query_suite_pd.iterrows():
        prop_id = row["property_id"]
        prop_cc = row["country_code"]
        dest_cc = row["dest_country"]
        print("\n---------------------------")
        print(f"Retrieving {n_candidates} candidates for query n. {i+1}...")
        print(f"prop_id: {prop_id}, prop_cc: {prop_cc}, dest_cc: {dest_cc}")
        t0 = time.perf_counter()
        cand_df = retrieve(
            target_id=prop_id,
            country=dest_cc,
            df=df_emb,
            lsh_model=lsh_model,
            n=n_candidates
        )

        i_suite_results = (
            cand_df
            .withColumn("target_id", F.lit(prop_id))
            .withColumn("target_cc", F.lit(prop_cc))
            .withColumn("cand_cc", F.lit(dest_cc))
        )

        if suite_results:
            suite_results = suite_results.union(i_suite_results)
        else:
            suite_results = i_suite_results

        t1 = time.perf_counter()
        print(f"[QSM] Retrieved candidates in {(t1 - t0):.2f}s!")

    suite_results = (   
        suite_results
        .select(['property_id', 'cand_cc', 'target_id', 'target_cc', 'features_norm', 'hashes', 'l2_dist'])
        .withColumnRenamed('property_id', 'cand_id')
        .withColumn('cosine_similarity', 1 - (F.col('l2_dist') ** 2) / 2.0)
    )

    # Join suite results with data
    suite_results = suite_results.join(df_all, on=[suite_results['cand_id']==df_all['property_id']], how='inner')

    print("\n[QSM] Finished retrieval.")

    return suite_results

def build_MCQS_and_save_results(df_all_path, df_emb_path, lsh_model_path, MCQS_out_path, MCQS_data_path, MCQS_results_out_path):
    """
    Build a multiple countries query suite (MCQS), save the suite, the associated data, and the retrieval results to disk.

    :param df_all_path: Path to the full property data parquet file
    :param df_emb_path: Path to the property embeddings parquet file
    :param lsh_model_path: Path to the trained LSH model
    :param MCQS_out_path: Output path for the MCQS query suite parquet
    :param MCQS_data_path: Output path for the MCQS data parquet
    :param MCQS_results_out_path: Output path for the MCQS retrieval results parquet
    """

    # Load the dataframes and lsh model
    t0 = time.perf_counter()

    df_emb = (
        spark.read.parquet(df_emb_path)
        .select("property_id", "addr_cc", "features_norm")
        .dropDuplicates(["property_id"])
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    emb_cnt = df_emb.count()

    df_all = (
        spark.read.parquet(df_all_path)
        .dropDuplicates(["property_id"])
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    all_cnt = df_all.count()

    lsh_model = BucketedRandomProjectionLSHModel.load(lsh_model_path)

    t1 = time.perf_counter()
    print(f"[QSM] Loaded df_emb rows={emb_cnt:,}, df_all rows={all_cnt:,} in {(t1 - t0):.2f}s")

    # Use the 3 least and 3 most frequent country codes as destinations in the query suite
    smallest_ccs, largest_ccs = pick_smallest_and_largest_ccs(df_all)

    print(f"[QSM] Smallest CCs: {smallest_ccs}")
    print(f"[QSM] Largest CCs: {largest_ccs}")

    dest_ccs = smallest_ccs + largest_ccs


    # Build the multiple countries query suite
    query_suite, targets_data = build_MCQS(df_all, df_emb, lsh_model, n_props=N_PROPS_MCQS, countries=dest_ccs)

    print("[QSM] Resulted multiple countries query suite:")
    print(query_suite)
    (
        spark.createDataFrame(query_suite)
        .write
        .mode("overwrite")
        .parquet(MCQS_out_path)
    )
    print(f"[QSM] Saved multiple countries QUERY SUITE to {MCQS_out_path}")
    
    targets_data.write.mode("overwrite").parquet(MCQS_data_path)
    print(f"[QSM] Saved multiple countries query suite DATA to {MCQS_data_path}")

    # Retrieve the MCQS results
    suite_results = retrieve_suite_results(df_all, df_emb, lsh_model, query_suite)
    suite_results.write.mode("overwrite").parquet(MCQS_results_out_path)
    print(f"[QSM] Saved multiple countries query suite RETRIEVAL RESULTS to {MCQS_results_out_path}")
    print("[QSM] Done!")

# ---- Run ----
build_MCQS_and_save_results(
    df_all_path=FULL_PATH,
    df_emb_path=EMBEDDED_PATH,
    lsh_model_path=LSH_MODEL_PATH,
    MCQS_out_path=MCQS_PATH,
    MCQS_data_path=MCQS_DATA_PATH,
    MCQS_results_out_path=MCQS_RESULTS_PATH
)