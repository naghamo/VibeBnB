from airbnb_load import *
from config import *
from pyspark.sql import functions as F
def osm_join(continents):
    osm_df = spark.read.parquet("dbfs:/FileStore/osm/south_america_pois_enriched.parquet")

    airbnb_df =airbnb_load()
    

    airbnb_scope = (
        airbnb
        .filter(F.col("country").isin(continents["South America"]))
        .select(
            "property_id",
            F.col("lat").cast("double").alias("lat"),
            F.col("long").cast("double").alias("lon"),
            "country"
        )
        .filter(F.col("lat").isNotNull() & F.col("lon").isNotNull())
    )

    # OSM POIs 
    osm_geo = (
        osm_df
        .select(
            F.col("lat").cast("double").alias("p_lat"),
            F.col("lon").cast("double").alias("p_lon"),
            F.lower(F.trim(F.col("poi_group"))).alias("group")   # single string
        )
        .filter(
            F.col("p_lat").isNotNull() &
            F.col("p_lon").isNotNull() &
            F.col("group").isNotNull() &
            (F.col("group") != "")
        )
    )

    a = airbnb_scope.alias("a")
    p = osm_geo.alias("p")

    # longitude delta depends on latitude 
    delta_lon_expr = (R_M / (111000.0 * F.cos(F.radians(F.col("a.lat")))))


    # Candidate bbox join

    cand = a.join(
        p,
        (F.col("p.p_lat").between(F.col("a.lat") - DELTA_LAT, F.col("a.lat") + DELTA_LAT)) &
        (F.col("p.p_lon").between(F.col("a.lon") - delta_lon_expr, F.col("a.lon") + delta_lon_expr)),
        how="inner"
    )



    dist_m = 2 * EARTH_R * F.asin(F.sqrt(
        F.pow(F.sin((F.radians(F.col("p.p_lat") - F.col("a.lat"))) / 2), 2) +
        F.cos(F.radians(F.col("a.lat"))) * F.cos(F.radians(F.col("p.p_lat"))) *
        F.pow(F.sin((F.radians(F.col("p.p_lon") - F.col("a.lon"))) / 2), 2)
    ))

    cand = (
        cand
        .withColumn("distance_m", dist_m)
        .filter(F.col("distance_m") <= R_M)
    )


    per_group = (
        cand
        .groupBy(F.col("a.property_id").alias("property_id"), F.col("p.group").alias("group"))
        .agg(F.count("*").cast("int").alias("n_places"))
    )

    env_json = (
        per_group
        .withColumn("val", F.struct(F.col("n_places").alias("n_places")))
        .groupBy("property_id")
        .agg(F.map_from_entries(F.collect_list(F.struct(F.col("group"), F.col("val")))).alias("env_group_map"))
        .withColumn("env_group_json", F.to_json(F.col("env_group_map")))
        .select("property_id", "env_group_json")
    )
    return env_json