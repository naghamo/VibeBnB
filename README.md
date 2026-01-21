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

### Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

