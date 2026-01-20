"""
Airbnb Property Data Loader and Preprocessor
=============================================

This module loads and preprocesses Airbnb property data,
preparing it for similarity-based recommendation systems using property embeddings.

The pipeline performs:
1. Data loading from Parquet files
2. Column selection and type casting
3. Geographic enrichment via reverse geocoding
4. Missing value analysis and imputation
5. Price normalization (EUR -> USD)
6. Statistical imputation using country-level and global fallbacks
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    IntegerType, ArrayType, FloatType, LongType
)
from pyspark.sql.functions import (
    col, when, coalesce, lit, broadcast, 
    from_json, map_from_entries, element_at,
    pandas_udf, round as spark_round, avg
)

import reverse_geocoder as rg

import re
import json


# =============================================================================
# Configuration Constants
# =============================================================================

class AirbnbConfig:
    """Configuration for Airbnb data loading pipeline."""
    
    # Azure Storage Configuration
    STORAGE_ACCOUNT = "lab94290"
    CONTAINER = "airbnb"
    DATA_PATH = "airbnb_1_12_parquet"
    SAS_TOKEN="sp=rle&st=2025-12-24T17:37:04Z&se=2026-02-28T01:52:04Z&spr=https&sv=2024-11-04&sr=c&sig=a0lx%2BS6PuS%2FvJ9Tbt4NKdCJHLE9d1Y1D6vpE1WKFQtk%3D".lstrip('?')
    
    # Currency Conversion Rates (EUR to USD)
    EUR_TO_USD_RATE = 1.08
    
    # Column Groups for Organization
    CORE_COLUMNS = [
        "property_id", "listing_name", "listing_title",
        "lat", "long"
    ]
    
    RATING_COLUMNS = [
        "ratings", "reviews", "property_number_of_reviews"
    ]
    
    HOST_COLUMNS = [
        "host_rating", "host_number_of_reviews", "host_response_rate",
        "hosts_year", "is_supperhost", "is_guest_favorite"
    ]
    
    PROPERTY_COLUMNS = [
        "guests", "category", "category_rating"
    ]
    
    TEXT_COLUMNS = [
        "amenities", "description", "description_items", 
        "details"
    ]
    
    PRICING_COLUMNS = [
        "pricing_details", "total_price", "currency"
    ]
    
    AVAILABILITY_COLUMNS = [
        "availability", "final_url"
    ]
    
    # Columns to impute using country/global statistics (or something else if needed)
    IMPUTABLE_COLUMNS = [
        "host_rating", "host_response_rate", "hosts_year",
        "rating_cleanliness", "rating_accuracy", "rating_checkin",
        "rating_communication", "rating_location", "rating_value",
        "price_per_night"
    ]

    DERIVED_COLUMNS = [
        "room_type_text", "n_beds", "n_baths", "n_bedrooms",
        "amenities_has_text", "amenities_no_text", "amenities_text", "custom_text_blob"
    ]


# =============================================================================
# Schema Definitions
# =============================================================================

def get_pricing_schema() -> StructType:
    """
    Define schema for pricing_details JSON blob.
    
    Returns:
        StructType for parsing pricing information
        
    Structure:
        - airbnb_service_fee: Platform fee
        - cleaning_fee: One-time cleaning cost
        - initial_price_per_night: Listed nightly rate
        - num_of_nights: Booking duration
        - price_per_night: Actual nightly rate (may differ from initial)
        - price_without_fees: Base price before extras
        - special_offer: Discount applied (negative value)
        - taxes: Local taxes
    """
    return StructType([
        StructField("airbnb_service_fee", DoubleType()),
        StructField("cleaning_fee", DoubleType()),
        StructField("initial_price_per_night", DoubleType()),
        StructField("num_of_nights", IntegerType()),
        StructField("price_per_night", DoubleType()),
        StructField("price_without_fees", DoubleType()),
        StructField("special_offer", DoubleType()),
        StructField("taxes", DoubleType())
    ])


def get_category_rating_schema() -> ArrayType:
    """
    Define schema for category_rating JSON blob.
    
    Returns:
        ArrayType containing rating categories
        
    Structure:
        Array of {name: str, value: str} for ratings like:
        - Cleanliness
        - Accuracy
        - Check-in
        - Communication
        - Location
        - Value
    """
    return ArrayType(StructType([
        StructField("name", StringType()),
        StructField("value", StringType())  # String initially, cast to double later
    ]))


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

def get_listing_features_schema():
    """Define schema for listing features extracted"""
    return StructType([
        StructField("room_type_text", StringType()),
        StructField("n_beds", DoubleType()),
        StructField("n_baths", DoubleType()),
        StructField("n_bedrooms", DoubleType()),
        StructField("amenities_has_text", StringType()),
        StructField("amenities_no_text", StringType()),
        StructField("amenities_text", StringType()),
        StructField("custom_text_blob", StringType()) 
    ])


# =============================================================================
# Data Loading
# =============================================================================

def configure_azure_storage(spark: SparkSession) -> None:
    """
    Configure Spark session for Azure Blob Storage access using SAS token.
    
    Args:
        spark: Active SparkSession
    """
    sas_token = AirbnbConfig.SAS_TOKEN
    storage_account = AirbnbConfig.STORAGE_ACCOUNT
    
    spark.conf.set(f"fs.azure.account.auth.type.{storage_account}.dfs.core.windows.net", "SAS")
    spark.conf.set(f"fs.azure.sas.token.provider.type.{storage_account}.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider")
    spark.conf.set(f"fs.azure.sas.fixed.token.{storage_account}.dfs.core.windows.net", sas_token)


def load_raw_airbnb_data(spark: SparkSession) -> DataFrame:
    """
    Load raw Airbnb data from Azure Blob Storage.
    
    Args:
        spark: Active SparkSession
        
    Returns:
        Raw Airbnb DataFrame
    """
    configure_azure_storage(spark)
    container = AirbnbConfig.CONTAINER
    storage_account = AirbnbConfig.STORAGE_ACCOUNT
    data_path = AirbnbConfig.DATA_PATH

    path = f"abfss://{container}@{storage_account}.dfs.core.windows.net/{data_path}"
    
    print(f"Loading Airbnb data from: {path}")
    df = spark.read.parquet(path)
    print(f"Loaded {df.count():,} rows")
    
    return df



# =============================================================================
# Column Selection and Type Casting
# =============================================================================

def select_relevant_columns(df: DataFrame) -> DataFrame:
    """
    Select columns relevant for property similarity analysis.
    
    Rationale:
        - Core identifiers and text for basic property info
        - Geographic coordinates for location-based features
        - Ratings for quality signals
        - Host metrics for trust/reliability signals
        - Property details for capacity and amenities
        - Pricing for affordability comparison
        
    Args:
        df: Raw Airbnb DataFrame
        
    Returns:
        DataFrame with selected columns
    """
    all_columns = (
        AirbnbConfig.CORE_COLUMNS +
        AirbnbConfig.RATING_COLUMNS +
        AirbnbConfig.HOST_COLUMNS +
        AirbnbConfig.PROPERTY_COLUMNS +
        AirbnbConfig.TEXT_COLUMNS +
        AirbnbConfig.PRICING_COLUMNS +
        AirbnbConfig.AVAILABILITY_COLUMNS
    )
    
    return df.select(*all_columns)


def cast_and_clean_types(df: DataFrame) -> DataFrame:
    """
    Cast columns to appropriate types and perform initial cleaning.
    
    Type Casting Strategy:
        - Numeric ratings/counts -> double/int
        - Boolean flags -> int (0/1) for embedding compatibility
        - Geographic coords -> double, filter out nulls (required for geocoding)
        - Remove duplicate property_ids (keep first occurrence)
        
    Args:
        df: DataFrame with selected columns
        
    Returns:
        DataFrame with proper types and deduplicated
    """
    df_cast = df \
        .withColumn("lat", col("lat").cast("double")) \
        .withColumn("long", col("long").cast("double")) \
        .filter(col("lat").isNotNull() & col("long").isNotNull()) \
        .withColumn("ratings", col("ratings").cast("double")) \
        .withColumn("property_number_of_reviews", col("property_number_of_reviews").cast("int")) \
        .withColumn("host_rating", col("host_rating").cast("double")) \
        .withColumn("host_number_of_reviews", col("host_number_of_reviews").cast("int")) \
        .withColumn("host_response_rate", col("host_response_rate").cast("double")) \
        .withColumn("hosts_year", col("hosts_year").cast("int")) \
        .withColumn("total_price", col("total_price").cast("double")) \
        .withColumn("guests", col("guests").cast("int")) \
        .withColumn("is_supperhost", (col("is_supperhost") == "true").cast("int")) \
        .withColumn("is_guest_favorite", (col("is_guest_favorite") == "true").cast("int")) \
        .withColumn("is_available", (col("availability") == "true").cast("int")) \
        .dropDuplicates(["property_id"])
    
    return df_cast


# =============================================================================
# JSON Parsing and Complex Type Handling
# =============================================================================

def parse_pricing_details(df: DataFrame) -> DataFrame:
    """
    Parse and flatten pricing_details JSON blob.
    
    Pricing Structure:
        - Extracts all fee components (service, cleaning, taxes)
        - Promotes fields to top-level columns for easier access
        - Preserves null values for later imputation
        
    Args:
        df: DataFrame with pricing_details column
        
    Returns:
        DataFrame with flattened pricing columns
    """
    pricing_schema = get_pricing_schema()
    
    df_parsed = df \
        .withColumn("pricing_struct", from_json(col("pricing_details"), pricing_schema)) \
        .select("*", "pricing_struct.*") \
        .drop("pricing_struct", "pricing_details")
    
    return df_parsed


def parse_category_ratings(df: DataFrame) -> DataFrame:
    """
    Parse and extract category_rating JSON array into individual rating columns.
    
    Category Ratings:
        - Converts array of {name, value} structs to a map
        - Extracts specific ratings (Cleanliness, Accuracy, etc.)
        - Casts string values to doubles
        
    Rationale:
        These granular ratings provide more signal than overall rating alone,
        useful for matching users who prioritize specific aspects (e.g., cleanliness)
        
    Args:
        df: DataFrame with category_rating column
        
    Returns:
        DataFrame with individual rating columns
    """
    category_schema = get_category_rating_schema()
    
    df_parsed = df \
        .withColumn("category_struct", from_json(col("category_rating"), category_schema)) \
        .withColumn("ratings_map", map_from_entries(col("category_struct"))) \
        .withColumn("rating_cleanliness", element_at(col("ratings_map"), "Cleanliness").cast("double")) \
        .withColumn("rating_accuracy", element_at(col("ratings_map"), "Accuracy").cast("double")) \
        .withColumn("rating_checkin", element_at(col("ratings_map"), "Check-in").cast("double")) \
        .withColumn("rating_communication", element_at(col("ratings_map"), "Communication").cast("double")) \
        .withColumn("rating_location", element_at(col("ratings_map"), "Location").cast("double")) \
        .withColumn("rating_value", element_at(col("ratings_map"), "Value").cast("double")) \
        .drop("category_struct", "ratings_map", "category_rating")
    
    return df_parsed


# =============================================================================
# Geographic Enrichment
# =============================================================================

def initialize_reverse_geocoder() -> None:
    """
    Initialize reverse geocoder by performing a dummy lookup.
    
    This warms up the geocoder's internal data structures for faster batch processing.
    """
    print("Initializing reverse geocoder...")
    rg.search((0, 0))
    print("Reverse geocoder ready!")



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


def add_geographic_enrichment(df: DataFrame) -> DataFrame:
    """
    Enrich properties with geographic metadata via reverse geocoding.
    
    Geographic Enrichment:
        - Converts (lat, long) to human-readable location info
        - Adds country, state/province, city for:
          * Country-level imputation statistics
          * Location-based filtering
          * Human-readable property context
        - Uses addr_ prefix to match POI data format
        
    Args:
        df: DataFrame with lat/long columns
        
    Returns:
        DataFrame with added addr_name, addr_cc, addr_admin1, addr_admin2 columns
    """
    initialize_reverse_geocoder()
    reverse_geocode_udf = create_reverse_geocode_udf()
    
    df_geo = df.withColumn(
        "geo_data",
        reverse_geocode_udf(col("lat"), col("long"))
    ).select(
        "*",
        col("geo_data.name").alias("addr_name"),
        col("geo_data.cc").alias("addr_cc"),
        col("geo_data.admin1").alias("addr_admin1"),
        col("geo_data.admin2").alias("addr_admin2")
    ).drop("geo_data")
    
    return df_geo

# =============================================================================
# Listing features cleaning
# =============================================================================

def create_listing_features_udf():
    listing_features_schema = get_listing_features_schema()

    @pandas_udf(listing_features_schema)
    def listing_features_udf(amenities_series: pd.Series, desc_series: pd.Series, details_series: pd.Series) -> pd.DataFrame:
        
        # Setup Limits for custom_text_blob
        AMENITY_LIMIT = 40
        DESC_LIMIT = 10
        DETAILS_LIMIT = 10
        
        # Setup Regex
        re_beds = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:bed|beds|betten)\b", re.IGNORECASE)
        re_baths = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:bath|baths|bathroom|bathrooms|badezimmer|bäder|baeder)\b", re.IGNORECASE)
        re_bedrooms = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:bedroom|bedrooms|schlafzimmer)\b", re.IGNORECASE)
        re_studio = re.compile(r"\bstudio\b", re.IGNORECASE)
        
        results = []
        
        for amen_json, desc_json, det_json in zip(amenities_series, desc_series, details_series):
            
            # Separate text blobs to match her logic
            desc_text = ""  # For beds/baths extraction
            details_text = ""  # For bedrooms extraction
            
            room_type = ""  # Default to empty string, not None
            has_list = []
            no_list = []
            
            custom_parts = []
            amenity_count = 0
            
            # ==========================================
            # A. PARSE DESCRIPTION_ITEMS
            # ==========================================
            if desc_json and isinstance(desc_json, str) and desc_json.startswith("["):
                try:
                    d_items = json.loads(desc_json)
                    if d_items and len(d_items) > 0:
                        # Room type is first item, lowercase and trimmed
                        room_type = str(d_items[0]).lower().strip()
                        
                        # Rest of items (index 1 onwards) joined with " | " for regex search
                        if len(d_items) > 1:
                            desc_text = " | ".join([str(x) for x in d_items[1:]])
                        
                        # Add to custom text with limit (first DESC_LIMIT items)
                        custom_parts.extend([str(x) for x in d_items[:DESC_LIMIT] if x])
                except:
                    pass

            # ==========================================
            # B. PARSE DETAILS
            # ==========================================
            if det_json and isinstance(det_json, str) and det_json.startswith("["):
                try:
                    det_items = json.loads(det_json)
                    if det_items:
                        # Join with " | " for regex search
                        details_text = " | ".join([str(x) for x in det_items])
                        
                        # Add to custom text with limit
                        custom_parts.extend([str(x) for x in det_items[:DETAILS_LIMIT] if x])
                except:
                    pass
                
            # ==========================================
            # C. PARSE AMENITIES
            # ==========================================
            if amen_json and isinstance(amen_json, str) and amen_json.startswith("["):
                try:
                    groups = json.loads(amen_json)
                    for group in groups:
                        if amenity_count >= AMENITY_LIMIT:
                            break
                        
                        g_name = str(group.get('group_name', '')).lower().strip()
                        # Check if group is "not included"
                        is_excluded = (g_name == "not included")
                        
                        for item in group.get('items', []):
                            if amenity_count >= AMENITY_LIMIT:
                                break
                            
                            name = item.get('name', '')
                            val = str(item.get('value', '')) if item.get('value') else ''
                            
                            # Skip if no name
                            if not name or not name.strip():
                                continue
                            
                            # Tokenize: lowercase, replace non-alphanumeric with _, trim underscores
                            clean_token = re.sub(r"[^a-z0-9]+", "_", name.lower().strip())
                            clean_token = re.sub(r"^_+|_+$", "", clean_token)
                            
                            # Check if this amenity is "not included"
                            is_no = is_excluded or val.startswith("SYSTEM_NO_")
                            
                            if is_no:
                                no_list.append(f"no_{clean_token}")
                            else:
                                has_list.append(clean_token)
                            
                            # Add to custom text blob (only non-excluded amenities)
                            if not is_no:
                                is_system_val = val.startswith("SYSTEM_")
                                
                                if not is_system_val:
                                    # Keep "name: value" format
                                    if val and val.lower() not in ['true', '1', '']:
                                        custom_parts.append(f"{name}: {val}")
                                    else:
                                        custom_parts.append(name)
                                else:
                                    custom_parts.append(name)
                                
                                amenity_count += 1
                except:
                    pass

            # ==========================================
            # D. EXTRACT NUMBERS
            # ==========================================
            # Beds/baths from description_items text (lowercase)
            desc_lower = desc_text.lower()
            beds_match = re_beds.search(desc_lower)
            baths_match = re_baths.search(desc_lower)
            
            # Bedrooms from details text (lowercase)
            details_lower = details_text.lower()
            bedrooms_match = re_bedrooms.search(details_lower)
            is_studio = re_studio.search(details_lower)
            
            n_beds = float(beds_match.group(1).replace(',', '.')) if beds_match else 0.0
            n_baths = float(baths_match.group(1).replace(',', '.')) if baths_match else 0.0
            n_bedrooms = 0.0 if is_studio else (float(bedrooms_match.group(1).replace(',', '.')) if bedrooms_match else 0.0)

            # ==========================================
            # E. BUILD RESULT
            # ==========================================
            # Remove duplicates while preserving order, filter out short items
            unique_custom = list(dict.fromkeys([p for p in custom_parts if p and len(p) > 2]))
            
            # Sort and deduplicate amenity lists
            amenities_has = " | ".join(sorted(set(has_list)))
            amenities_no = " | ".join(sorted(set(no_list)))
            
            # Combined amenities text (has + no, separated by " | ")
            amenities_text = (amenities_has + " | " + amenities_no).strip(" | ")
            
            results.append({
                "room_type_text": room_type,  # Empty string if not found
                "n_beds": n_beds,
                "n_baths": n_baths,
                "n_bedrooms": n_bedrooms,
                "amenities_has_text": amenities_has,
                "amenities_no_text": amenities_no,
                "amenities_text": amenities_text,
                "custom_text_blob": ", ".join(unique_custom)
            })
            
        return pd.DataFrame(results)
    
    return listing_features_udf

def apply_listing_feature_extraction(df: DataFrame) -> DataFrame:
    """
    Extracts structured features (beds, baths) and text blobs from raw JSONs.
    """
    listing_features_udf = create_listing_features_udf()
    return df.withColumn("feats", listing_features_udf(col("amenities"), col("description_items"), col("details"))) \
             .select("*", "feats.*") \
             .drop("feats")

# =============================================================================
# Missing Value Analysis
# =============================================================================

def analyze_missing_values(df: DataFrame, show_all: bool = False) -> pd.DataFrame:
    """
    Analyze and report missing values in the DataFrame.
    
    Args:
        df: PySpark DataFrame to analyze
        show_all: If True, show all columns; if False, only show columns with missing values
        
    Returns:
        Styled pandas DataFrame with missing value statistics
    """
    print("Analyzing missing values...")
    
    missing_counts = df.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in df.columns
    ]).toPandas()
    
    total_count = df.count()
    missing_df = pd.DataFrame({
        'column': missing_counts.columns,
        'missing': missing_counts.iloc[0].values,
    })
    missing_df['missing_pct'] = (missing_df['missing'] / total_count * 100).round(2)
    missing_df['present'] = total_count - missing_df['missing']
    missing_df['present_pct'] = (missing_df['present'] / total_count * 100).round(2)
    
    missing_df = missing_df.sort_values('missing_pct', ascending=False)
    
    if not show_all:
        missing_df = missing_df[missing_df['missing'] > 0]
    
    missing_df = missing_df.reset_index(drop=True)
    
    cols_with_missing = (missing_df['missing'] > 0).sum()
    print(f"\nTotal rows: {total_count:,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Columns with missing values: {cols_with_missing}\n")
    
    return missing_df


# =============================================================================
# Missing Value Handling
# =============================================================================

def drop_rows_with_critical_nulls(df: DataFrame) -> Tuple[DataFrame, int]:
    """
    Drop rows missing fields that have <1% missing rate.
    
    Fields:
        - property_id: Required identifier
        - listing_name, listing_title: Essential for text embeddings
        - ratings, is_guest_favorite: Core quality signals
        - guests: Capacity information
        - amenities, description, details: Text features
        
    Rationale:
        These fields have <1% missing rate and are important for creating
        meaningful property representations. No reason to impute them.
        
    Args:
        df: DataFrame to filter
        
    Returns:
        Tuple of (filtered DataFrame, number of rows dropped)
    """
    initial_count = df.count()
    
    df_filtered = df.filter(
        col("property_id").isNotNull() &
        col("listing_name").isNotNull() &
        col("listing_title").isNotNull() &
        col("ratings").isNotNull() &
        col("is_guest_favorite").isNotNull() &
        col("guests").isNotNull() &
        col("amenities").isNotNull() &
        col("details").isNotNull() &
        col("description_items").isNotNull()
    )
    
    dropped = initial_count - df_filtered.count()
    print(f"Dropped {dropped:,} rows with critical missing values")
    
    return df_filtered, dropped


def fill_simple_defaults(df: DataFrame) -> DataFrame:
    """
    Fill missing values with reasonable defaults for simple numeric/categorical fields.
    
    Default Strategy:
        - Review counts -> 0 (no reviews yet)
        - Boolean flags -> 0 (false/not applicable)
        - Optional fees -> 0.0 (no fee charged)
        - num_of_nights -> 1 (base nightly rate)
        
    Rationale:
        These are zero-inflated features where missing typically means
        "not applicable" or "none" rather than unknown information.
        
    Args:
        df: DataFrame with missing values
        
    Returns:
        DataFrame with simple defaults filled
    """
    df_filled = df \
        .withColumn("property_number_of_reviews", coalesce(col("property_number_of_reviews"), lit(0))) \
        .withColumn("host_number_of_reviews", coalesce(col("host_number_of_reviews"), lit(0))) \
        .withColumn("is_supperhost", coalesce(col("is_supperhost"), lit(0))) \
        .withColumn("is_available", coalesce(col("is_available"), lit(0))) \
        .withColumn("cleaning_fee", coalesce(col("cleaning_fee"), lit(0.0))) \
        .withColumn("airbnb_service_fee", coalesce(col("airbnb_service_fee"), lit(0.0))) \
        .withColumn("num_of_nights", coalesce(col("num_of_nights"), lit(1))) \
        .withColumn("taxes", coalesce(col("taxes"), lit(0.0))) \
        .withColumn("special_offer", coalesce(col("special_offer"), lit(0.0)))
    
    return df_filled


def fill_text_defaults(df: DataFrame) -> DataFrame:
    """
    Fill missing text fields with fallback values.
    
    Text Fallback Strategy:
        - description: Use listing_title → listing_name → placeholder
        
    Rationale:
        Having *some* text is better than null for embedding generation.
        The hierarchy ensures we use the most descriptive available text.
        
    Args:
        df: DataFrame with text columns
        
    Returns:
        DataFrame with text defaults filled
    """
    df_text = df.withColumn(
        "description",
        coalesce(
            col("description"),
            col("listing_title"),
            col("listing_name"),
            lit("No description provided")
        )
    )
    
    return df_text


# =============================================================================
# Price Derivation and Normalization
# =============================================================================

def derive_price_per_night(df: DataFrame) -> DataFrame:
    """
    Derive price_per_night from total_price when not explicitly provided.
    
    Price Derivation Logic:
        1. Reverse-engineer from total_price by subtracting all fees/discounts
        2. Use waterfall approach with multiple fallbacks
        3. Ensure no negative prices (data quality check)
        
    Formula:
        derived_price = (total - service_fee - cleaning - taxes - special_offer) / nights
        
    Waterfall Priority:
        1. price_per_night (explicit)
        2. initial_price_per_night (explicit)
        3. price_without_fees / nights (explicit base)
        4. derived_price_per_night (calculated)
        5. total_price / nights (last resort)
        
    Args:
        df: DataFrame with pricing columns
        
    Returns:
        DataFrame with final_price_per_night column
    """
    df_derived = df.withColumn(
        "derived_price_per_night",
        (col("total_price") - col("airbnb_service_fee") - col("cleaning_fee") 
         - col("taxes") - col("special_offer")) / col("num_of_nights")
    )
    
    # Replace original columns with final versions
    df_final = df_derived.withColumn(
        "price_per_night",  # Overwrite original
        coalesce(
            col("price_per_night"),
            col("initial_price_per_night"),
            col("price_without_fees") / col("num_of_nights"),
            col("derived_price_per_night"),
            col("total_price") / col("num_of_nights")
        )
    ).withColumn(
        "num_of_nights",  # Overwrite original
        col("num_of_nights")  # Already has safe default from fill_simple_defaults
    )
    
    # Clean up derived column
    df_final = df_final.drop("derived_price_per_night")
    
    # Ensure no negative prices
    df_final = df_final.withColumn(
        "price_per_night",
        when(col("price_per_night") < 0, col("total_price"))
        .otherwise(spark_round(col("price_per_night"), 2))
    )
    
    return df_final


def normalize_currency_to_usd(df: DataFrame) -> DataFrame:
    """
    Convert all prices to USD for cross-country comparability.
    
    Currency Normalization:
        - USD -> keep as-is
        - EUR -> multiply by conversion rate
        - Other currencies -> set to null (not enough data for reliable conversion)
        
    Rationale:
        Cross-country similarity requires common price scale.
        EUR and USD cover majority of dataset.
        
    Args:
        df: DataFrame with money_cols and currency columns
        
    Returns:
        DataFrame with normalized money_cols
    """

    # Define all columns that contain money
    money_cols = [
        "price_per_night", 
        # "total_price", # We only use price_per_night, others are unncessary (unless we chnage anything?)
        # "cleaning_fee", 
        # "airbnb_service_fee", 
        # "taxes", 
        # "special_offer",
        # "initial_price_per_night",
        # "price_without_fees"
    ]
    
    df_usd = df
    
    # Loop through and convert EUR -> USD
    for c in money_cols:
        if c in df_usd.columns:
            df_usd = df_usd.withColumn(
                c,
                when(col("currency") == "EUR", col(c) * AirbnbConfig.EUR_TO_USD_RATE)
                .otherwise(col(c))
            ).withColumn(c, spark_round(col(c), 2))
            
    # Standardize currency label
    df_usd = df_usd.withColumn(
        "currency", 
        when(col("currency") == "EUR", "USD").otherwise(col("currency"))
    )
    
    return df_usd

def filter_price_outliers(df: DataFrame, max_price: float = 1000.0) -> DataFrame:
    """
    Filter out rows where price_per_night exceeds a threshold (e.g., $1000).
    """
    initial_count = df.count()
    
    # Keep rows where price is LESS than or EQUAL to max_price
    df_filtered = df.filter(col("price_per_night") <= max_price)
    
    dropped = initial_count - df_filtered.count()
    print(f"Dropped {dropped:,} rows with price > ${max_price}")
    
    return df_filtered


# =============================================================================
# Statistical Imputation
# =============================================================================

def compute_imputation_statistics(df: DataFrame) -> Tuple[DataFrame, Dict[str, float]]:
    """
    Compute country-level and global statistics for imputation.
    
    Statistical Imputation Strategy:
        - Country-level averages: Accounts for regional differences
          (e.g., ratings norms, typical host tenure, price ranges vary by market)
        - Global fallbacks: Used when country data is missing/sparse
        
    Computed For:
        - Host metrics (rating, response rate, tenure)
        - Category ratings (cleanliness, accuracy, etc.)
        - Price per night
        
    Args:
        df: DataFrame with properties to impute
        
    Returns:
        Tuple of (country statistics DataFrame, global defaults dict)
    """
    imputable_cols = AirbnbConfig.IMPUTABLE_COLUMNS
    
    # Country-level statistics
    agg_exprs = [spark_round(avg(c), 2).alias(f"avg_{c}") for c in imputable_cols]
    country_stats = df.groupBy("addr_cc").agg(*agg_exprs)
    
    # Global statistics
    global_stats_row = df.agg(*agg_exprs).collect()[0]
    global_defaults = {c: global_stats_row[f"avg_{c}"] for c in imputable_cols}
    
    print("Computed imputation statistics:")
    print(f"  - Country groups: {country_stats.count()}")
    print(f"  - Global defaults: {len(global_defaults)} columns")
    
    return country_stats, global_defaults


def apply_statistical_imputation(
    df: DataFrame,
    country_stats: DataFrame,
    global_defaults: Dict[str, float]
) -> DataFrame:
    """
    Impute missing values using country-level and global statistics.
    
    Imputation Waterfall (per column):
        1. Original value (if present)
        2. Country average (if available)
        3. Global average (final fallback)
        
    Rationale:
        This approach balances specificity (country context) with
        robustness (global fallback) to avoid dropping valuable properties.
        
    Args:
        df: DataFrame with missing values
        country_stats: Country-level averages (from compute_imputation_statistics)
        global_defaults: Global averages (from compute_imputation_statistics)
        
    Returns:
        DataFrame with imputed values
    """
    imputable_cols = AirbnbConfig.IMPUTABLE_COLUMNS
    
    # Join country stats (broadcast for performance)
    df_joined = df.join(broadcast(country_stats), on="addr_cc", how="left")
    
    # Apply waterfall imputation for each column
    df_imputed = df_joined
    for c in imputable_cols:
        df_imputed = df_imputed.withColumn(
            c,
            coalesce(
                col(c),                     # 1. Original value
                col(f"avg_{c}"),            # 2. Country average
                lit(global_defaults[c])     # 3. Global average
            )
        ).drop(f"avg_{c}")  # Clean up temporary column
    
    return df_imputed


# =============================================================================
# Final Column Selection
# =============================================================================

def select_final_columns(df: DataFrame) -> DataFrame:
    """
    Select and order final columns for downstream embedding pipeline.
    
    Column Organization:
        - Identifiers (property_id, names)
        - Geographic metadata (lat, long, addr_*)
        - Quality signals (ratings, reviews)
        - Host signals (rating, tenure, badges)
        - Property details (category, specific ratings)
        - Text features (amenities, description, details)
        - Pricing (normalized per-night rate, capacity)
        - Availability metadata
        
    Args:
        df: Fully processed DataFrame
        
    Returns:
        DataFrame with final column selection
    """
    final_cols = [
        # Identifiers
        "property_id", "listing_name", "listing_title",
        
        # Geographic
        "lat", "long", "addr_name", "addr_admin1", "addr_cc",
        
        # Overall quality metrics
        "ratings", "property_number_of_reviews",
        
        # Host signals
        "host_rating", "host_number_of_reviews", "host_response_rate",
        "hosts_year", "is_supperhost", "is_guest_favorite",
        
        # Category-specific ratings
        "category", "rating_cleanliness", "rating_accuracy",
        "rating_checkin", "rating_communication", "rating_location", "rating_value",
        
        # Text features
        "amenities", "description", "description_items", "details",
        
        # Pricing (normalized)
        "price_per_night", "num_of_nights", "guests",
        
        # Availability
        "is_available", "final_url"
    ] + AirbnbConfig.DERIVED_COLUMNS

    log_cols = [c for c in df.columns if c.startswith("log_1p_")]
    
    # Combine the lists
    return df.select(*(final_cols + log_cols))


# =============================================================================
# Main Pipeline
# =============================================================================

def load_airbnb_data(spark: SparkSession) -> DataFrame:
    """
    Main pipeline: Load and preprocess Airbnb data for similarity analysis.
    
    Pipeline Steps:
        1. Load raw data from Azure
        2. Select relevant columns
        3. Cast types and deduplicate + extract listing features
        4. Parse JSON structures (pricing, ratings)
        5. Enrich with geographic metadata
        6. Analyze missing values
        7. Drop rows with critical nulls
        8. Fill simple defaults
        9. Derive and normalize prices
        10. Compute imputation statistics
        11. Apply statistical imputation
        12. Fill text defaults
        13. Select final columns
        
    Args:
        spark: Active SparkSession
        
    Returns:
        Cleaned and processed DataFrame ready for embedding generation
        
    Example:
        >>> spark = SparkSession.builder.appName("AirbnbLoader").getOrCreate()
        >>> df = load_airbnb_data(spark)
        >>> df.show(5)
    """
    print("=" * 80)
    print("Starting Airbnb Data Loading Pipeline")
    print("=" * 80)
    
    # Step 1: Load raw data
    df = load_raw_airbnb_data(spark)
    
    # Step 2-3: Select and cast
    print("\nStep 2-3: Selecting columns and casting types...")
    df = select_relevant_columns(df)
    df = cast_and_clean_types(df)
    print(f"After casting: {df.count():,} rows")

    print("\nStep 3.5: Extracting listing features from text...")
    df = apply_listing_feature_extraction(df)
    
    # Step 4: Parse complex structures
    print("\nStep 4: Parsing JSON structures...")
    df = parse_pricing_details(df)
    df = parse_category_ratings(df)
    
    # Step 5: Geographic enrichment
    print("\nStep 5: Adding geographic enrichment...")
    df = add_geographic_enrichment(df)
    
    # Step 6: Analyze missing values
    print("\nStep 6: Analyzing missing values...")
    missing_report = analyze_missing_values(df)
    print(missing_report.to_string())
    
    # Step 7: Drop critical nulls
    print("\nStep 7: Dropping rows with critical missing values...")
    df, _ = drop_rows_with_critical_nulls(df)
    
    # Step 8: Fill simple defaults
    print("\nStep 8: Filling simple default values...")
    df = fill_simple_defaults(df)
    
    # Step 9: Price derivation and normalization
    print("\nStep 9: Deriving and normalizing prices...")
    df = derive_price_per_night(df)
    df = normalize_currency_to_usd(df)
    
    # Step 10-11: Statistical imputation
    print("\nStep 10-11: Computing and applying statistical imputation...")
    country_stats, global_defaults = compute_imputation_statistics(df)
    df = apply_statistical_imputation(df, country_stats, global_defaults)
    
    # Step 12: Text defaults
    print("\nStep 12: Filling text defaults...")
    df = fill_text_defaults(df)
    
    # ==========================================================
    # ==========================================================
    print("\nStep 12.5: Filtering price outliers...")
    # This filters OUT rows > k (keeps rows <= k)
    df = filter_price_outliers(df, max_price=1300.0)

    print("\nStep 12.5.5: Applying Log1p Normalization...") # lol
    # Define the columns you want to normalize (usually skewed numeric data)
    cols_to_log = ["price_per_night", "property_number_of_reviews" "host_number_of_reviews"]
    df = apply_log1p_transform(df, cols_to_log)
    # ==========================================================
    # ==========================================================

    # Step 13: Final selection
    print("\nStep 13: Selecting final columns...")
    df = select_final_columns(df)
    
    print("\n" + "=" * 80)
    print(f"Pipeline Complete! Final dataset: {df.count():,} rows, {len(df.columns)} columns")
    print("=" * 80)
    
    return df


# =============================================================================
# Utility Functions
# =============================================================================

def save_processed_data(df: DataFrame, output_path: str, mode: str = "overwrite") -> None:
    """
    Save processed DataFrame to Parquet format.
    
    Args:
        df: Processed DataFrame to save
        output_path: Destination path (local or cloud storage)
        mode: Write mode ('overwrite', 'append', etc.)
    """
    print(f"\nSaving processed data to: {output_path}")
    df.write.mode(mode).parquet(output_path)
    print("Save complete!")

def apply_log1p_transform(df: DataFrame, columns: List[str]) -> DataFrame:
    """
    Applies log1p transformation (log(x + 1)) to specified columns.
    Creates new columns named 'log_1p_{col_name}'.
    
    Args:
        df: Input DataFrame
        columns: List of column names to transform
        
    Returns:
        DataFrame with new log-transformed columns
    """
    df_log = df
    for col_name in columns:
        # Check if column exists to avoid errors
        if col_name in df.columns:
            df_log = df_log.withColumn(f"log_1p_{col_name}", F.log1p(col(col_name)))
        else:
            print(f"Warning: Column '{col_name}' not found, skipping log transform.")
            
    return df_log

def analyze_distributions(
    df: DataFrame,
    numerical_cols: Optional[List[str]] = None,
    text_cols: Optional[List[str]] = None,
    iqr_multiplier: float = 1.5,
    show_samples: bool = True,
    skip_all_null_cols: bool = True,
) -> pd.DataFrame:
    """
    Optimized distribution analysis: 2-pass approach (1 stats job + 1 outlier job).
    Robust to columns that are entirely null (percentile_approx returns None).
    """

    # Auto-detect Numerical Columns
    if numerical_cols is None:
        numerical_cols = []
        for field in df.schema.fields:
            if isinstance(field.dataType, (DoubleType, IntegerType, FloatType, LongType)):
                if not field.name.startswith("is_") and "id" not in field.name.lower():
                    numerical_cols.append(field.name)
        print(f"Auto-detected {len(numerical_cols)} numerical columns")

    # Auto-detect Text Columns
    if text_cols is None:
        text_cols = []
        for field in df.schema.fields:
            if isinstance(field.dataType, StringType):
                if field.name not in ["addr_cc", "currency", "category", "cancellation_policy"]:
                    text_cols.append(field.name)
        print(f"Auto-detected {len(text_cols)} text columns")

    # Create checking DataFrame with length columns
    len_cols = [F.length(F.col(c)).alias(f"len_{c}") for c in text_cols]
    df_check = df.select(*numerical_cols, *len_cols)

    all_cols = numerical_cols + [f"len_{c}" for c in text_cols]
    total_count = df_check.count()

    print(f"\nAnalyzing {len(all_cols)} features ({total_count:,} rows)...")

    # PASS 1: Compute statistics (+ non-null counts)
    print("Pass 1: Computing global statistics...")
    stats_exprs = []
    for c in all_cols:
        stats_exprs.extend([
            F.count(F.col(c)).alias(f"{c}_nn"),
            F.min(c).alias(f"{c}_min"),
            F.max(c).alias(f"{c}_max"),
            F.avg(c).alias(f"{c}_mean"),
            F.stddev(c).alias(f"{c}_std"),
            # If percentile_approx returns null (all values null), coalesce to [None, None, None]
            F.coalesce(
                F.percentile_approx(c, [0.25, 0.5, 0.75], 5000),
                F.array(F.lit(None), F.lit(None), F.lit(None))
            ).alias(f"{c}_qs")
        ])

    stats_row = df_check.agg(*stats_exprs).first()

    # Decide which columns are actually analyzable
    nn_by_col = {c: int(stats_row[f"{c}_nn"]) for c in all_cols}
    valid_cols = [c for c in all_cols if nn_by_col[c] > 0]
    null_only_cols = [c for c in all_cols if nn_by_col[c] == 0]

    if null_only_cols:
        msg = f"Found {len(null_only_cols)} all-null columns: {null_only_cols}"
        if skip_all_null_cols:
            print(msg + " -> skipping")
        else:
            print(msg + " -> will include (outliers=0, stats=None)")

    cols_for_outliers = valid_cols if skip_all_null_cols else all_cols

    # PASS 2: Count outliers (batched) only for valid cols
    print("Pass 2: Checking outlier counts...")
    outlier_exprs = []
    fences = {}

    for c in cols_for_outliers:
        q25, median, q75 = stats_row[f"{c}_qs"]

        # If any quartile is None (should only happen when nn==0, but safe anyway)
        if q25 is None or q75 is None:
            fences[c] = (None, None)
            outlier_exprs.append(F.lit(0).alias(f"{c}_outliers"))
            continue

        iqr = q75 - q25
        if iqr == 0 or iqr is None:
            iqr = 0.01  # tiny value to avoid 0-width fences

        lower = q25 - (iqr_multiplier * iqr)
        upper = q75 + (iqr_multiplier * iqr)
        fences[c] = (lower, upper)

        is_outlier = (F.col(c) > upper) | (F.col(c) < lower)
        outlier_exprs.append(F.sum(F.when(is_outlier, 1).otherwise(0)).alias(f"{c}_outliers"))

    outlier_counts = df_check.agg(*outlier_exprs).first() if outlier_exprs else None

    # BUILD REPORT
    results_data = []
    outlier_details = []

    cols_for_report = valid_cols if skip_all_null_cols else all_cols

    for c in cols_for_report:
        mn, mx = stats_row[f"{c}_min"], stats_row[f"{c}_max"]
        avg, std = stats_row[f"{c}_mean"], stats_row[f"{c}_std"]
        q25, median, q75 = stats_row[f"{c}_qs"]

        lower, upper = fences.get(c, (None, None))

        if outlier_counts is not None and f"{c}_outliers" in outlier_counts.asDict():
            o_count = int(outlier_counts[f"{c}_outliers"])
        else:
            o_count = 0

        o_pct = (o_count / total_count * 100) if total_count else 0.0
        label = c

        # IMPORTANT: use `is not None` so 0 doesn't become None
        results_data.append({
            "Feature": label,
            "NonNull": nn_by_col[c],
            "Min": round(mn, 2) if mn is not None else None,
            "Q25": round(q25, 2) if q25 is not None else None,
            "Median": round(median, 2) if median is not None else None,
            "Mean": round(avg, 2) if avg is not None else None,
            "Q75": round(q75, 2) if q75 is not None else None,
            "Max": round(mx, 2) if mx is not None else None,
            "Lower Fence": round(lower, 2) if lower is not None else None,
            "Upper Fence": round(upper, 2) if upper is not None else None
        })

    pdf_results = pd.DataFrame(results_data)
    # display(pdf_results)
    return pdf_results


if __name__ == "__main__":
    # Example usage
    print(__doc__)
    print("\nThis module is designed to be imported and used in a notebook or script.")
    print("\nExample usage:")
    print("""
    from airbnb_data_loader import load_airbnb_data
    
    spark = SparkSession.builder.getOrCreate()
    
    df = load_airbnb_data(spark)
    df.show(10)
    """)






# from pyspark.sql.functions import col, regexp_replace
# from pyspark.sql.functions import col, split, trim, lower, size
# from pyspark.sql import SparkSession
# from spark_session import get_spark

# spark = get_spark()

# def airbnb_select(airbnb):
#     airbnb_sel = airbnb.select(
#     "property_id",
#     "listing_name",
 
#     "lat",
#     "long",
#     "location",

#     "ratings",
#     # "reviews",
#     "property_number_of_reviews",

#     "host_rating",
#     "host_number_of_reviews",
#     "host_response_rate",
#     "hosts_year",
#     "is_supperhost",
#     "is_guest_favorite",

#     "guests",

#     "category",
#     "category_rating",

#     "amenities",
#     "description",
#     "description_items",
#     "details",
#     # "arrangement_details",

#     "pricing_details",
#     "total_price",
#     "currency",
#     # "discount",

#     "availability",
#     "final_url"
#     )
#     return airbnb_sel

# def airbnb_clean(airbnb_sel):
#     return airbnb_sel.withColumn("lat", col("lat").cast("double")).withColumn("long", col("long").cast("double")).filter(col("lat").isNotNull() & col("long").isNotNull())\
#     .withColumn("ratings", col("ratings").cast("double")).withColumn("property_number_of_reviews",col("property_number_of_reviews").cast("int")).withColumn("host_rating", col("host_rating").cast("double")).withColumn("host_number_of_reviews", col("host_number_of_reviews").cast("int")).withColumn("host_response_rate",col("host_response_rate").cast("double")).withColumn("hosts_year", col("hosts_year").cast("int")).withColumn("total_price",col("total_price").cast("double")).withColumn("guests", col("guests").cast("int"))\
#         .withColumn("is_supperhost", (col("is_supperhost") == "true").cast("int")).withColumn("is_guest_favorite", (col("is_guest_favorite") == "true").cast("int")).withColumn("is_available",(col("availability") == "true").cast("int"))\
#             .withColumn("city", trim(split(col("location"), ",").getItem(0))).withColumn("country", trim(split(col("location"), ",").getItem(size(split(col("location"), ",")) - 1)))\
#         .dropDuplicates(["property_id"])

    
# def airbnb_load():
#     spark = SparkSession.builder.getOrCreate()

#     storage_account = "lab94290"  
#     container = "airbnb"
#     sas_token="sp=rle&st=2025-12-24T17:37:04Z&se=2026-02-28T01:52:04Z&spr=https&sv=2024-11-04&sr=c&sig=a0lx%2BS6PuS%2FvJ9Tbt4NKdCJHLE9d1Y1D6vpE1WKFQtk%3D"
#     sas_token = sas_token.lstrip('?')
#     spark.conf.set(f"fs.azure.account.auth.type.{storage_account}.dfs.core.windows.net", "SAS")
#     spark.conf.set(f"fs.azure.sas.token.provider.type.{storage_account}.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider")
#     spark.conf.set(f"fs.azure.sas.fixed.token.{storage_account}.dfs.core.windows.net", sas_token)
#     path = f"abfss://{container}@{storage_account}.dfs.core.windows.net/airbnb_1_12_parquet"
#     airbnb = spark.read.parquet(path)
#     airbnb_sel = airbnb_select(airbnb)
#     airbnb_clean_df = airbnb_clean(airbnb_sel)
#     return airbnb_clean_df





