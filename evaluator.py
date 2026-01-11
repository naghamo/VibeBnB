from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel
from pyspark.ml.feature import BucketedRandomProjectionLSHModel
from retrieve_rank import retrieve, order
from config import *
import time
import random
import pandas as pd
import warnings


EUROPE_CC = set(continents['europe'])

def build_query_suite(df_all, df_emb, n_queries):
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

    # country value counts df will be used to search for suitable queries from all data
    country_value_counts = (
    df_all.select("addr_cc")
          .groupBy("addr_cc")
          .count()
          .toPandas()
    )

    print("Initializing variables...")
    cc_pool = countries_europe.copy()
    queries = pd.DataFrame(columns=['property_id','country_code'])
    seed_country = random.choice(cc_pool)
    cc_pool.remove(seed_country)

    print("Picking the first seed property...")
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
    print(f"Found initial seed property in {(t_1 - t_0):.2f}s.")
    print(f"Starting query suite building process from property {seed_prop_id} in {seed_country}!")

    queries = queries.append({'property_id': seed_prop_id, 'country_code': seed_country}, ignore_index=True)

    for i in range(n_queries-1):
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

    dest_countries = random.sample(cc_pool, k=n_queries)
    queries['dest_country'] = dest_countries

    print("\nFinished.")

    return queries


def retrieve_suite_results(df_emb, query_suite):

    # Retrieve top 50 for each query, and save it to a dataframe
    query_suite_pd = query_suite.toPandas()
    suite_results = None
    n_candidates = 50
    print(f"Starting retrieval for {len(query_suite_pd)} queries...")

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
        print(f"Retrieved candidates in {(t1 - t0):.2f}s!")

    suite_results = (   
        suite_results
        .select(['property_id', 'cand_cc', 'target_id', 'target_cc', 'features_norm', 'hashes', 'l2_dist'])
        .withColumnRenamed('property_id', 'cand_id')
        .withColumn('cosine_similarity', 1 - (F.col('l2_dist') ** 2) / 2.0)
    )

    print("\nFinished.")

    return suite_results