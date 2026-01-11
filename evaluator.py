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

    return queries