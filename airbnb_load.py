from pyspark.sql.functions import col, regexp_replace
from pyspark.sql.functions import col, split, trim, lower, size
from pyspark.sql import SparkSession
from spark_session import get_spark

spark = get_spark()

def airbnb_select(airbnb):
    airbnb_sel = airbnb.select(
    "property_id",
    "listing_name",
 
    "lat",
    "long",
    "location",

    "ratings",
    # "reviews",
    "property_number_of_reviews",

    "host_rating",
    "host_number_of_reviews",
    "host_response_rate",
    "hosts_year",
    "is_supperhost",
    "is_guest_favorite",

    "guests",

    "category",
    "category_rating",

    "amenities",
    "description",
    "description_items",
    "details",
    # "arrangement_details",

    "pricing_details",
    "total_price",
    "currency",
    # "discount",

    "availability",
    "final_url"
    )
    return airbnb_sel

def airbnb_clean(airbnb_sel):
    return airbnb_sel.withColumn("lat", col("lat").cast("double")).withColumn("long", col("long").cast("double")).filter(col("lat").isNotNull() & col("long").isNotNull())\
    .withColumn("ratings", col("ratings").cast("double")).withColumn("property_number_of_reviews",col("property_number_of_reviews").cast("int")).withColumn("host_rating", col("host_rating").cast("double")).withColumn("host_number_of_reviews", col("host_number_of_reviews").cast("int")).withColumn("host_response_rate",col("host_response_rate").cast("double")).withColumn("hosts_year", col("hosts_year").cast("int")).withColumn("total_price",col("total_price").cast("double")).withColumn("guests", col("guests").cast("int"))\
        .withColumn("is_supperhost", (col("is_supperhost") == "true").cast("int")).withColumn("is_guest_favorite", (col("is_guest_favorite") == "true").cast("int")).withColumn("is_available",(col("availability") == "true").cast("int"))\
            .withColumn("city", trim(split(col("location"), ",").getItem(0))).withColumn("country", trim(split(col("location"), ",").getItem(size(split(col("location"), ",")) - 1)))\
        .dropDuplicates(["property_id"])

    
def airbnb_load():
    spark = SparkSession.builder.getOrCreate()

    storage_account = "lab94290"  
    container = "airbnb"
    sas_token="sp=rle&st=2025-12-24T17:37:04Z&se=2026-02-28T01:52:04Z&spr=https&sv=2024-11-04&sr=c&sig=a0lx%2BS6PuS%2FvJ9Tbt4NKdCJHLE9d1Y1D6vpE1WKFQtk%3D"
    sas_token = sas_token.lstrip('?')
    spark.conf.set(f"fs.azure.account.auth.type.{storage_account}.dfs.core.windows.net", "SAS")
    spark.conf.set(f"fs.azure.sas.token.provider.type.{storage_account}.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider")
    spark.conf.set(f"fs.azure.sas.fixed.token.{storage_account}.dfs.core.windows.net", sas_token)
    path = f"abfss://{container}@{storage_account}.dfs.core.windows.net/airbnb_1_12_parquet"
    airbnb = spark.read.parquet(path)
    airbnb_sel = airbnb_select(airbnb)
    airbnb_clean_df = airbnb_clean(airbnb_sel)
    return airbnb_clean_df





