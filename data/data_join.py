# data_join.py
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel
import time
from config import *
from data.airbnb_data_loader import load_airbnb_data
from data.travel_cities_data_loader import load_travel_cities


def cities_airbnb_join(cities_df: DataFrame, airbnb_df: DataFrame) -> DataFrame:
    """
    Join Airbnb listings with city-level metadata using (addr_cc, addr_name).

    Notes:
    - Uses a broadcast join for the city table (typically much smaller than listings).
    - Renames city addr_* fields that might collide with Airbnb columns.
    - Keeps all Airbnb rows (left join), adding city metadata when available.
    """

    # Rename cities columns that may collide with Airbnb location fields.
    c = (cities_df
         .withColumnRenamed("addr_admin1", "city_addr_admin1")
         .withColumnRenamed("addr_admin2", "city_addr_admin2"))

    # Join on country code + city name (reverse-geocoded identifiers).
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

    # Drop duplicate join keys from the cities side (Airbnb keys remain).
    joined = joined.drop(F.col("c.addr_cc")).drop(F.col("c.addr_name"))

    return joined


def _sanitize(c: str) -> str:
    """Make column-safe names for env feature columns (letters/digits/underscore)."""
    return "".join(ch if ch.isalnum() else "_" for ch in c)


def osm_join_one_continent(continent: str, airbnb_df: DataFrame, osm_df: DataFrame) -> DataFrame:
    """
    Compute per-listing neighborhood environment counts from OSM POIs for one continent.

    Output:
    - One row per Airbnb listing with env_* count columns (pivoted by POI group).
    - Counts represent POIs within a radius R_M around the listing (Haversine distance).
    """
    # Scope Airbnb to the requested continent using the configured country-code list.
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

    # OSM input is expected to already be filtered to relevant geography before calling.
    # Keep only POI coordinates + the semantic group label used for aggregation.
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

    # Longitude window depends on latitude (accounts for meridian convergence).
    # This is used only for a cheap bounding-box prefilter before Haversine.
    delta_lon = (R_M / (111000.0 * F.cos(F.radians(F.col("a.lat")))))

    # Candidate pairs via bounding box prefilter (reduces distance computations).
    cand = (
        airbnb_scope.alias("a")
        .join(
            osm_geo.alias("p"),
            (F.col("p.p_lat").between(F.col("a.lat") - DELTA_LAT, F.col("a.lat") + DELTA_LAT))
            & (F.col("p.p_lon").between(F.col("a.lon") - delta_lon, F.col("a.lon") + delta_lon)),
            "inner",
        )
    )

    # Exact Haversine distance (meters) for filtered candidate pairs.
    dist = 2 * EARTH_R * F.asin(
        F.sqrt(
            F.pow(F.sin(F.radians(F.col("p.p_lat") - F.col("a.lat")) / 2), 2)
            + F.cos(F.radians(F.col("a.lat")))
            * F.cos(F.radians(F.col("p.p_lat")))
            * F.pow(F.sin(F.radians(F.col("p.p_lon") - F.col("a.lon")) / 2), 2)
        )
    )

    # Aggregate POI counts within radius by (listing, POI group).
    per_group = (
        cand.withColumn("distance_m", dist)
        .filter(F.col("distance_m") <= R_M)
        .groupBy(F.col("a.property_id").alias("property_id"), F.col("p.group").alias("group"))
        .agg(F.count("*").cast("int").alias("n_places"))
    )

    # Pivot to wide format: one env column per POI group (missing => 0).
    pivoted = per_group.groupBy("property_id").pivot("group").agg(F.first("n_places")).fillna(0)

    # Rename pivoted group columns into stable env_* feature names.
    renamed = pivoted.select(
        "property_id",
        *[F.col(c).alias(f"env_{_sanitize(c)}") for c in pivoted.columns if c != "property_id"],
    )

    # Attach environment features back to the scoped Airbnb dataframe.
    out = airbnb_scope.join(renamed, "property_id", "left")

    # For listings with no nearby POIs in a category, env_* should be 0 (not null).
    env_cols = [c for c in out.columns if c.startswith("env_")]
    if env_cols:
        out = out.fillna(0, subset=env_cols)

    # Optional: enforce a fixed env schema if ENV_COLS is defined in config/globals.
    # This ensures downstream pipelines see all expected env columns.
    if "ENV_COLS" in globals():
        out = out.select("*", *[F.lit(0).alias(c) for c in ENV_COLS if c not in out.columns])

    return out


def join_europe_by_country_and_save(
    out_countries_dir: str,
    cities_path: str = "dbfs:/vibebnb/data/travel_cities.parquet",
    osm_base_dir = "dbfs:/vibebnb/data/osm_pois",
) -> None:
    """
    Build the enriched Europe dataset country-by-country and save partitions.

    For each European country (cc):
      1) Filter Airbnb, cities, and OSM to that country.
      2) Join city-level metadata into Airbnb (left join).
      3) Compute OSM environment counts within radius and attach env_* features.
      4) Join city-enriched Airbnb with env_* features on property_id.
      5) Save as parquet to out_countries_dir/country=CC.

    This country-wise processing limits spatial joins to smaller subsets.
    """
    spark = SparkSession.builder.getOrCreate()
    europe_cc = continents["europe"]
    total_countries = len(europe_cc)

    # Load once and cache (reused across country loop).
    cities_df_all = load_travel_cities(spark, cities_path).persist(StorageLevel.MEMORY_AND_DISK)
    airbnb_all = load_airbnb_data(spark).dropDuplicates(["property_id"]).persist(StorageLevel.MEMORY_AND_DISK)
    osm_europe = spark.read.parquet(f"{osm_base_dir}/europe_pois_enriched.parquet").persist(StorageLevel.MEMORY_AND_DISK)

    # Materialize caches once to avoid repeated lineage recomputation.
    _ = cities_df_all.count()
    _ = airbnb_all.count()
    _ = osm_europe.count()

    print(f"[EU] Starting Europe processing: {total_countries} countries")

    for idx, cc in enumerate(europe_cc, start=1):
        t0 = time.perf_counter()
        print(f"\n[EU] ({idx}/{total_countries}) Country: {cc} | START")

        # Filter all sources to a single country code.
        airbnb_cc = airbnb_all.filter(F.col("addr_cc") == cc)
        cities_cc = cities_df_all.filter(F.col("addr_cc") == cc)
        osm_cc = osm_europe.filter(F.col("addr_cc") == cc)

        # Join city-level metadata into Airbnb.
        airbnb_with_cities = cities_airbnb_join(cities_cc, airbnb_cc)
        # airbnb_with_cities.printSchema()

        # Compute and attach env_* features from OSM (drop location fields from this side).
        env_cc = osm_join_one_continent("europe", airbnb_cc, osm_cc).drop("lat", "lon", "addr_cc","addr_admin1","addr_admin2","addr_name")

        # Final join: city-enriched Airbnb + environment features.
        final_cc = airbnb_with_cities.join(env_cc, "property_id", "left")

        # Ensure env_* columns are zero-filled for missing neighborhoods.
        env_cols = [c for c in final_cc.columns if c.startswith("env_")]
        if env_cols:
            final_cc = final_cc.fillna(0, subset=env_cols)

        # Save per-country parquet partition.
        country_path = f"{out_countries_dir}/country={cc}"
        print(f"[EU] ({idx}/{total_countries}) Saving -> {country_path}")
        final_cc.write.mode("overwrite").parquet(country_path)

        t1 = time.perf_counter()
        print(f"[EU] ({idx}/{total_countries}) Country: {cc} | DONE in {(t1 - t0):.2f} sec")

    # Cleanup cached DataFrames.
    cities_df_all.unpersist()
    airbnb_all.unpersist()
    osm_europe.unpersist()

    print("\n[EU] Done. Saved per-country datasets only.")


# join_europe_by_country_and_save(
#     out_countries_dir="dbfs:/vibebnb/data/europe_countries_joined_",
# )
