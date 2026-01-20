# data_join.py
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel
import time
from travel_cities_data_loader import load_travel_cities
from config import *   
from airbnb_data_loader import load_airbnb_data
from travel_cities_data_loader import load_travel_cities


def cities_airbnb_join(cities_df: DataFrame, airbnb_df: DataFrame) -> DataFrame:
    """
    Join Airbnb listings with city metadata using (addr_cc, addr_name).
    Avoid duplicate columns by renaming/dropping cities addr_* fields after the join.
    """

    #rename cities addr_* that collide with Airbnb
    c = (cities_df
         .withColumnRenamed("addr_admin1", "city_addr_admin1")
         .withColumnRenamed("addr_admin2", "city_addr_admin2"))

    # 2) join on country+city name (you already do that)
    joined = (
        airbnb_df.alias("a")
        .join(
            F.broadcast(c.alias("c")),
            on=[
                F.col("a.addr_cc") == F.col("c.addr_cc"),
                F.col("a.addr_name") == F.col("c.addr_name"),
            ],
            how="left",
        )
    )

    # drop join-duplicate keys from cities side (keep Airbnb versions)
    joined = joined.drop(F.col("c.addr_cc")).drop(F.col("c.addr_name"))

    return joined


def _sanitize(c: str) -> str:
    """Make column-safe names for env features."""
    return "".join(ch if ch.isalnum() else "_" for ch in c)


def osm_join_one_continent(continent: str, airbnb_df: DataFrame, osm_df: DataFrame) -> DataFrame:
    """Compute per-listing environment counts from OSM for one continent."""
    airbnb_scope = (
        airbnb_df
        .filter(F.col("addr_cc").isin(continents[continent]))
        .select(
            "property_id",
            F.col("lat").cast("double").alias("lat"),
            F.col("long").cast("double").alias("lon"),
            "addr_cc",
        )
        .filter(F.col("lat").isNotNull() & F.col("lon").isNotNull())
        .dropDuplicates(["property_id"])
    )

    # OSM already filtered to the relevant country before calling this function
    osm_geo = (
        osm_df
        .select(
            F.col("lat").cast("double").alias("p_lat"),
            F.col("lon").cast("double").alias("p_lon"),
            F.lower(F.trim(F.col("poi_group"))).alias("group"),
        )
        .filter(
            F.col("p_lat").isNotNull()
            & F.col("p_lon").isNotNull()
            & F.col("group").isNotNull()
            & (F.col("group") != "")
        )
    )

    delta_lon = (R_M / (111000.0 * F.cos(F.radians(F.col("a.lat")))))

    cand = (
        airbnb_scope.alias("a")
        .join(
            osm_geo.alias("p"),
            (F.col("p.p_lat").between(F.col("a.lat") - DELTA_LAT, F.col("a.lat") + DELTA_LAT))
            & (F.col("p.p_lon").between(F.col("a.lon") - delta_lon, F.col("a.lon") + delta_lon)),
            "inner",
        )
    )

    dist = 2 * EARTH_R * F.asin(
        F.sqrt(
            F.pow(F.sin(F.radians(F.col("p.p_lat") - F.col("a.lat")) / 2), 2)
            + F.cos(F.radians(F.col("a.lat")))
            * F.cos(F.radians(F.col("p.p_lat")))
            * F.pow(F.sin(F.radians(F.col("p.p_lon") - F.col("a.lon")) / 2), 2)
        )
    )

    per_group = (
        cand.withColumn("distance_m", dist)
        .filter(F.col("distance_m") <= R_M)
        .groupBy(F.col("a.property_id").alias("property_id"), F.col("p.group").alias("group"))
        .agg(F.count("*").cast("int").alias("n_places"))
    )

    pivoted = per_group.groupBy("property_id").pivot("group").agg(F.first("n_places")).fillna(0)

    renamed = pivoted.select(
        "property_id",
        *[F.col(c).alias(f"env_{_sanitize(c)}") for c in pivoted.columns if c != "property_id"],
    )

    out = airbnb_scope.join(renamed, "property_id", "left")

    env_cols = [c for c in out.columns if c.startswith("env_")]
    if env_cols:
        out = out.fillna(0, subset=env_cols)

    if "ENV_COLS" in globals():
        out = out.select("*", *[F.lit(0).alias(c) for c in ENV_COLS if c not in out.columns])

    return out


def join_europe_by_country_and_save(
    out_countries_dir: str,
    cities_path: str = "dbfs:/vibebnb/data/travel_cities.parquet",
    osm_base_dir = "dbfs:/vibebnb/data/osm_pois",
) -> None:
    """
    Europe country-by-country:
      - For each country: join cities + join OSM + save that country parquet
      - Prints progress as (idx/total) and prints time spent per country join+save
    """
    spark = SparkSession.builder.getOrCreate()
    europe_cc = continents["europe"]
    total_countries = len(europe_cc)

    # Load once + cache
    cities_df_all = load_travel_cities(spark, cities_path).persist(StorageLevel.MEMORY_AND_DISK)
    airbnb_all = load_airbnb_data(spark).dropDuplicates(["property_id"]).persist(StorageLevel.MEMORY_AND_DISK)
    osm_europe = spark.read.parquet(f"{osm_base_dir}/europe_pois_enriched.parquet").persist(StorageLevel.MEMORY_AND_DISK)

    # Materialize caches once
    _ = cities_df_all.count()
    _ = airbnb_all.count()
    _ = osm_europe.count()

    print(f"[EU] Starting Europe processing: {total_countries} countries")

    for idx, cc in enumerate(europe_cc, start=1):
        t0 = time.perf_counter()
        print(f"\n[EU] ({idx}/{total_countries}) Country: {cc} | START")

        # Filter by country
        airbnb_cc = airbnb_all.filter(F.col("addr_cc") == cc)
        cities_cc = cities_df_all.filter(F.col("addr_cc") == cc)
        osm_cc = osm_europe.filter(F.col("addr_cc") == cc)

        # Join cities
        airbnb_with_cities = cities_airbnb_join(cities_cc, airbnb_cc)
        # airbnb_with_cities.printSchema()

        # Join OSM 
        env_cc = osm_join_one_continent("europe", airbnb_cc, osm_cc).drop("lat", "lon", "addr_cc","addr_admin1","addr_admin2","addr_name")

        # Final join
        final_cc = airbnb_with_cities.join(env_cc, "property_id", "left")

        env_cols = [c for c in final_cc.columns if c.startswith("env_")]
        if env_cols:
            final_cc = final_cc.fillna(0, subset=env_cols)

        # Save
        country_path = f"{out_countries_dir}/country={cc}"
        print(f"[EU] ({idx}/{total_countries}) Saving -> {country_path}")
        final_cc.write.mode("overwrite").parquet(country_path)

        t1 = time.perf_counter()
        print(f"[EU] ({idx}/{total_countries}) Country: {cc} | DONE in {(t1 - t0):.2f} sec")

    # Cleanup
    cities_df_all.unpersist()
    airbnb_all.unpersist()
    osm_europe.unpersist()

    print("\n[EU] Done. Saved per-country datasets only.")


join_europe_by_country_and_save(
    out_countries_dir="dbfs:/vibebnb/data/europe_countries_joined",
)
