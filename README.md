# VibeBnB 🏡  
A Similarity-Based, Preference-Aware Recommendation System for Airbnb Listings

---

## Overview

**VibeBnB** is a data-driven recommendation system designed to help users discover
Airbnb listings with a similar *“vibe”* across different cities and countries.
Unlike standard filter-based search, the system models similarity using a
combination of:

- intrinsic listing attributes (price, quality, host reputation),
- neighborhood-level environmental context derived from OpenStreetMap (OSM),
- optional city-level travel context (climate and budget).

The system follows a clear **offline / online architecture**:
- an **offline pipeline** performs all heavy computation (data integration,
  scoring, embeddings, indexing),
- an **online pipeline** supports fast similarity retrieval and
  preference-aware ranking at query time.

The design emphasizes **interpretability**, **scalability**, and **explicit user
control**.

---

## Data Sources

VibeBnB integrates three heterogeneous datasets:

### Airbnb Listings (Primary Dataset)
Provides listing-level information, including pricing, capacity, amenities,
host-related metadata, guest ratings, and geographic coordinates. This dataset
serves as the backbone of the recommendation system.

### OpenStreetMap (OSM) Points of Interest
Used to model neighborhood-level environmental context.
Each OSM item corresponds to a geolocated point of interest (POI) with semantic
tags describing its function (e.g., food, transport, leisure). POIs are grouped
into high-level categories to capture neighborhood “vibe”.

### City-Level Travel Data
Provides coarse, destination-level context such as monthly average temperatures,
budget classification, and tourism-related indicators. This dataset covers
approximately 560 major travel cities.

---

## System Architecture

The system is organized into two pipelines:

### Offline Pipeline
Executed once (or periodically) to prepare all data required for online querying.
It includes:
1. Data cleaning and preprocessing
2. Cross-source data integration
3. Computation of interpretable component scores
4. Hybrid embedding construction
5. Locality-Sensitive Hashing (LSH) index construction

All outputs (scores, embeddings, and index) are saved **partitioned by country** to
enable efficient online querying.

### Online Pipeline
Triggered by user interaction and consists of:
1. Approximate nearest-neighbor retrieval using the precomputed LSH index
2. Preference-aware ranking based on user-defined weights
3. Returning the top-*k* recommended listings

---

## How to Run

### Execution Environment

All components of VibeBnB are designed to run within a **Databricks environment**.
Both the offline and online pipelines operate on data stored in the
**Databricks File System (DBFS)**.

As a result:
- there is **no need to manually download or load raw datasets**,
- all required data artifacts (cleaned data, scores, embeddings, indices, and
  precomputed candidates) are already persisted and reused across runs.

---

## Running the Offline Pipeline

The full offline pipeline is executed through the notebook:

```
offline_pipeline_run.ipynb
```

To run the offline pipeline:
1. Clone the Repo into Databricks.
1. Open the notebook.
2. Run all cells **in order**.

The notebook performs all required stages, including data cleaning, data integration, score
computation, embedding construction, and LSH index building.  
All outputs are saved to DBFS and reused by the online components.

No additional setup or data loading is required.

---

## Running the Online Pipeline

The online pipeline can be explored through the demonstration notebook:

```
demo.ipynb
```

This notebook demonstrates the full online flow using precomputed artifacts
stored in DBFS:
- similarity-based retrieval using the LSH index,
- preference-aware ranking,
- inspection of recommendation results.

The demo notebook does **not** require rerunning the offline pipeline and can be
executed independently.

---

## Web Interface Prototype

A lightweight interactive prototype of the system is deployed and available at:

🔗 **https://vibebnb-983293358114278.18.azure.databricksapps.com/**

The interface operates entirely on precomputed data and does not require running
any notebooks or jobs in Databricks. Users can simply open the link and interact
with the system.

The prototype is intentionally limited in scope:
- it uses **500 randomly sampled reference listings**,
- recommendations are available for **five selected target countries**,
- for each reference listing and target country, **50 candidate listings** are
  precomputed and stored.

At runtime, the interface applies the same preference-aware ranking mechanism as
the full online pipeline to reorder the precomputed candidates according to user
preferences.

---

## Notes

- The system follows an **offline-heavy, online-light** design.
- All similarity computations are interpretable and reproducible.
- City-level context is excluded from embeddings and applied only at ranking time.
- The project can be extended to additional countries or larger datasets with
  minimal changes to the pipeline.

## Repository Structure

The repository is organized to clearly separate data processing, analysis,
offline computation, online retrieval, and application deployment:
 
*All the files marked with  $  are for implementing the prototype interface with Flask with the Databrickes App
 ```
.
├── data/                           # Data loading and integration logic
│   ├── airbnb_data_loader.py       # Airbnb listings loading and preprocessing
│   ├── travel_cities_data_loader.py# City-level travel data loading and preprocessing
│   └── data_join.py                # Cross-source data integration and joins
│
├── EDA/                            # Exploratory data analysis
│   ├── data_analysis/              # Analysis notebooks and scripts
│
├── eval/                           # Evaluation scripts and analysis utilities
│
├── static/                         # Static assets for the web interface  $ 
├── templates/                      # HTML templates for the Flask app  $ 
│
├── app.py                          # Flask application entry point  $ 
├── app.yaml                        # Databricks Apps deployment configuration  $ 
├── config.py                       # Global configuration and constants
│
├── offline_pipeline_run.ipynb      # Full offline pipeline execution notebook
├── demo.ipynb                      # End-to-end demo of the online pipeline
│
├── osm_extraction.py               # OpenStreetMap POI extraction and preprocessing
├── score.py                        # Interpretable score computation logic
├── embeddings_fit.py               # Embedding construction and LSH indexing
├── retrieve_rank.py                # Online retrieval and preference-aware ranking
│
├── requirements.txt                # Python dependencies  $ 
└── README.md                       # Project documentation


```

