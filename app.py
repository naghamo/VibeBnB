import os
from flask import Flask, render_template, request

app = Flask(__name__)

# --- Temporary dummy data  ---
DUMMY_LISTINGS = [
    {"id": 101, "name": "Cozy Studio", "city": "Barcelona", "country": "Spain", "price": 85, "rating": 4.7},
    {"id": 102, "name": "Family Apartment", "city": "Berlin", "country": "Germany", "price": 140, "rating": 4.6},
    {"id": 103, "name": "Modern Loft", "city": "Paris", "country": "France", "price": 170, "rating": 4.8},
    {"id": 104, "name": "Budget Room", "city": "Prague", "country": "Czechia", "price": 45, "rating": 4.2},
    {"id": 105, "name": "Sea View Flat", "city": "Lisbon", "country": "Portugal", "price": 120, "rating": 4.5},
]
# from pyspark.sql import SparkSession

# spark = SparkSession.builder.getOrCreate()


# storage_account = "lab94290"
# container = "airbnb"

# sas_token = "sp=rle&st=2025-12-05T10:01:31Z&se=2026-02-26T18:23:31Z&spr=https&sv=2024-11-04&sr=c&sig=zdjdrz3xY%2BYfIMQt8Dvg2tzzKMHNR4sXttO%2BTUhAvJM%3D"
# sas_token = sas_token.lstrip('?')

# spark.conf.set(
#     f"fs.azure.account.auth.type.{storage_account}.dfs.core.windows.net", "SAS"
# )
# spark.conf.set(
#     f"fs.azure.sas.token.provider.type.{storage_account}.dfs.core.windows.net",
#     "org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider"
# )
# spark.conf.set(
#     f"fs.azure.sas.fixed.token.{storage_account}.dfs.core.windows.net",
#     sas_token
# )

# path = f"abfss://{container}@{storage_account}.dfs.core.windows.net/airbnb_1_12_parquet"


# airbnb_df = spark.read.parquet(path)


# ui_df = (
#     airbnb_df
#     .select(
#         "listing_id",
#         "name",
#         # "city",
#         # "country",
#         # "total_price",
#         # "review_scores_rating"
#     )
#     .limit(300)   
# )

# # Convert to Pandas for Flask template
# LISTINGS_FOR_UI = ui_df.toPandas().to_dict(orient="records")

COUNTRIES = sorted({x["country"] for x in DUMMY_LISTINGS} | {"Italy", "Netherlands", "USA", "UK"})

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    submitted = None

    if request.method == "POST":
        # Read form values
        selected_listing_id = request.form.get("selected_listing_id", "").strip()
        target_country = request.form.get("target_country", "").strip()
        k = int(request.form.get("k", "5"))

        # Preferences (0-100)
        pref_environment = int(request.form.get("pref_environment", "50"))
        pref_host = int(request.form.get("pref_host", "50"))
        pref_property = int(request.form.get("pref_property", "50"))
        pref_cost = int(request.form.get("pref_cost", "50"))

        submitted = {
            "selected_listing_id": selected_listing_id,
            "target_country": target_country,
            "k": k,
            "prefs": {
                "environment": pref_environment,
                "host_quality": pref_host,
                "property_quality": pref_property,
                "cost": pref_cost,
            }
        }

        # Placeholder result (later: call recommender and display real results)
        if not selected_listing_id:
            result = "Please select a listing from the table before searching."
        else:
            result = (
                f"UI received: reference listing {selected_listing_id}, "
                f"target country={target_country}, k={k}, prefs={submitted['prefs']}."
            )

    return render_template(
        "index.html",
        listings=DUMMY_LISTINGS,
        countries=COUNTRIES,
        result=result,
        submitted=submitted
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))  
    app.run(host="0.0.0.0", port=port)
