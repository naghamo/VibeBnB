
from pyspark.sql import SparkSession, DataFrame
import reverse_geocoder as rg
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    IntegerType, ArrayType
)
import pandas as pd
from pyspark.sql.functions import (
    col, when, coalesce, lit, broadcast, 
    from_json, map_from_entries, element_at,
    pandas_udf, round as spark_round, avg
)
def get_geo_schema() -> StructType:
    """
    Define schema for reverse geocoding results.
    
    Returns:
        StructType for geographic metadata
        
    Structure:
        - name: City name
        - cc: Country code (ISO 2-letter)
        - admin1: State/Province
        - admin2: County/Region
    """
    return StructType([
        StructField("name", StringType()),
        StructField("cc", StringType()),
        StructField("admin1", StringType()),
        StructField("admin2", StringType()),
    ])
def create_reverse_geocode_udf():
    """
    Create a Pandas UDF for batch reverse geocoding.
    
    Geocoding Strategy:
        - Processes coordinates in batches for efficiency
        - Handles null/invalid coordinates gracefully
        - Uses mode=2 for faster performance (approximate results)
        
    Returns:
        Pandas UDF function for geocoding
    """
    geo_schema = get_geo_schema()
    
    @pandas_udf(geo_schema)
    def reverse_geocode_udf(lat_series: pd.Series, long_series: pd.Series) -> pd.DataFrame:
        """Batch reverse geocode lat/long coordinates"""
        lats = pd.to_numeric(lat_series, errors='coerce')
        longs = pd.to_numeric(long_series, errors='coerce')
        
        coords = []
        valid_indices = []
        for i, (lat, lon) in enumerate(zip(lats, longs)):
            if pd.notna(lat) and pd.notna(lon):
                coords.append((lat, lon))
                valid_indices.append(i)
        
        if coords:
            results = rg.search(coords, mode=2)
        else:
            results = []
        
        output = []
        result_idx = 0
        for i in range(len(lat_series)):
            if i in valid_indices:
                r = results[result_idx]
                output.append({
                    "name": r.get("name"),
                    "cc": r.get("cc"),
                    "admin1": r.get("admin1"),
                    "admin2": r.get("admin2"),
                })
                result_idx += 1
            else:
                output.append({
                    "name": None,
                    "cc": None,
                    "admin1": None,
                    "admin2": None,
                })
        
        return pd.DataFrame(output)
    
    return reverse_geocode_udf


    
def initialize_reverse_geocoder() -> None:
    """
    Initialize reverse geocoder by performing a dummy lookup.
    
    This warms up the geocoder's internal data structures for faster batch processing.
    """
    print("Initializing reverse geocoder...")
    rg.search((0, 0))
    print("Reverse geocoder ready!")

def add_city_geographic_enrichment(cities_df: DataFrame, lat_col: str = "lat", lon_col: str = "lon") -> DataFrame:
    """Add addr_* columns to the cities dataset using reverse geocoding (same schema as Airbnb)."""
    initialize_reverse_geocoder()
    reverse_geocode_udf = create_reverse_geocode_udf()
    return (
        cities_df
        .withColumn("geo_data", reverse_geocode_udf(col(lat_col), col(lon_col)))
        .select(
            "*",
            col("geo_data.name").alias("addr_name"),
            col("geo_data.cc").alias("addr_cc"),
            col("geo_data.admin1").alias("addr_admin1"),
            col("geo_data.admin2").alias("addr_admin2"),
        )
        .drop("geo_data")
    )


def load_travel_cities(spark,cities_path = ""):
        
    cities_df = spark.read.parquet(cities_path)
    return add_city_geographic_enrichment(cities_df, lat_col="latitude", lon_col="longitude")



