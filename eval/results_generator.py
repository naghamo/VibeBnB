!pip install joypy

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

    # Build a "wide" DF: one column per cand_cc holding cosine_similarity, NaN otherwise
    wide = suite_results_pd[["target_id", "cand_cc", "cosine_similarity"]].copy()
    for cc in ccs:
        wide[cc] = np.where(wide["cand_cc"].eq(cc), wide["cosine_similarity"], np.nan)

    wide = wide.drop(columns=["cand_cc", "cosine_similarity"])

    # Choose distinct colors (consistent per cand_cc)
    if got_smallest_and_largest:
        n_small = len(smallest_ccs)
        n_large = len(largest_ccs)
        blues = plt.cm.Blues(np.linspace(0.5, 0.9, n_small))
        oranges = plt.cm.Oranges(np.linspace(0.5, 0.9, n_large))
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
        by="target_id",
        column=ccs,
        color=colors,
        legend=True,
        overlap=2,
        linewidth=1,
        alpha=0.6,
        figsize=(8, 10),
        fade=False
    )

    plt.title("Ridgeline Plot: cosine_similarity by cand_cc (overlaid per target_id)")
    plt.show()