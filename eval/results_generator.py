#!pip install joypy

from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel
from pyspark.ml.feature import BucketedRandomProjectionLSHModel
from retrieve_rank import retrieve, order
from config import *
import matplotlib.pyplot as plt
import time
import random
import pandas as pd
import numpy as np
import warnings
import joypy

MCQS_data_pd = spark.read.parquet(MCQS_DATA_PATH).toPandas()

def truncate_label(label, max_len=15):
    """Truncates a label with an ellipsis if it exceeds max_len."""
    if len(label) > max_len:
        return label[:max_len-3] + '...'
    return label

def create_ridge_plot(suite_results_pd, smallest_ccs = [], largest_ccs = []):
    """
    Plots a ridge plot where each ridge is a property from the queries, and a kde plot for each country is plotted.

    :param suite_results_pd: the query suite results dataframe
    :param smallest_ccs: a list of less-data countries
    :param largest_ccs: a list of more-data countries
    """
    got_smallest_and_largest = len(smallest_ccs) and len(largest_ccs)
    if got_smallest_and_largest:
        ccs = smallest_ccs + largest_ccs
    else:
        ccs = sorted(suite_results_pd["cand_cc"].unique())

    # min_n = 20
    # counts = suite_results_pd.groupby(["target_id","cand_cc"]).size()
    # ok = counts[counts >= min_n].index
    # suite_results_pd = suite_results_pd.set_index(["target_id","cand_cc"]).loc[ok].reset_index()

    # Build a "wide" DF: one column per cand_cc holding cosine_similarity, NaN otherwise
    MCQS_data_pd["joyplot_label"] = MCQS_data_pd['listing_title'].apply(truncate_label, max_len=17)
    MCQS_data_pd["joyplot_label"] = MCQS_data_pd['addr_cc'] + ", " + MCQS_data_pd['joyplot_label']

    wide = suite_results_pd[["target_id", "cand_cc", "cosine_similarity"]].copy()
    for cc in ccs:
        wide[cc] = np.where(wide["cand_cc"].eq(cc), wide["cosine_similarity"], np.nan)

    wide = wide.drop(columns=["cand_cc", "cosine_similarity"])
    temp_df = MCQS_data_pd[["property_id", "joyplot_label"]]
    wide = wide.merge(temp_df, left_on="target_id", right_on="property_id", how="inner")


    # Choose distinct colors (consistent per cand_cc)
    if got_smallest_and_largest:
        n_small = len(smallest_ccs)
        n_large = len(largest_ccs)
        blues = plt.cm.viridis(np.linspace(0.0, 0.5, n_small))
        oranges = plt.cm.inferno(np.linspace(0.5, 0.9, n_large))
        colors = []
        for cc in ccs:
            if cc in smallest_ccs:
                idx = smallest_ccs.index(cc)
                colors.append(blues[idx])
            elif cc in largest_ccs:
                idx = largest_ccs.index(cc)
                colors.append(oranges[idx])
            else:
                colors.append("grey")
    else:
        colors = [plt.cm.tab10(i % 10) for i in range(len(ccs))]

    fig, axes = joypy.joyplot(
        wide,
        by="joyplot_label",
        column=ccs,
        color=colors,
        legend=True,
        overlap=2,
        linewidth=1,
        alpha=0.5,
        figsize=(8, 10),
        fade=False,
        ylim='own'
    )

    plt.suptitle("Cosine Similarity by Destination Country (overlaid per property)", y=1.02, fontweight='bold')
    plt.savefig("ridgeline.png", dpi=300)
