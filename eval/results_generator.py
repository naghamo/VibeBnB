#!pip install joypy

from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel
from pyspark.ml.feature import BucketedRandomProjectionLSHModel
from retrieve_rank import retrieve, order
from config import *

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import time
import random
import pandas as pd
import numpy as np
import warnings
import joypy

MCQS_data_pd = spark.read.parquet(MCQS_DATA_PATH).toPandas()


def truncate_label(label, max_len=15):
    """Truncates a label with an ellipsis if it exceeds max_len."""
    label = "" if label is None else str(label)
    if len(label) > max_len:
        return label[: max_len - 3] + "..."
    return label


def create_ridge_plot(
    suite_results_pd: pd.DataFrame,
    smallest_ccs=None,
    largest_ccs=None,
    out_path: str = "ridgeline.png",
):
    """
    Plots a ridge plot where each ridge is a property (target_id),
    and an overlaid KDE for each destination country (cand_cc) is plotted.

    Legend is drawn INSIDE the plot as a single box with two titled sections:
    "Small countries" and "Large countries" (no separator border between them).

    Adds axis labels:
      - X label on the bottom ridge axis
      - Figure-level Y label (since joypy uses multiple axes)
    """
    if smallest_ccs is None:
        smallest_ccs = []
    if largest_ccs is None:
        largest_ccs = []

    got_groups = (len(smallest_ccs) > 0) and (len(largest_ccs) > 0)
    if got_groups:
        ccs = list(smallest_ccs) + list(largest_ccs)
    else:
        ccs = sorted(suite_results_pd["cand_cc"].dropna().unique().tolist())

    # ---- Labels (once) ----
    MCQS_data_pd["joyplot_label"] = MCQS_data_pd["listing_title"].apply(
        truncate_label, max_len=17
    )

    # ---- Build "wide" DF: one column per cand_cc ----
    wide = suite_results_pd[["target_id", "cand_cc", "cosine_similarity"]].copy()
    for cc in ccs:
        wide[cc] = np.where(wide["cand_cc"].eq(cc), wide["cosine_similarity"], np.nan)

    wide = wide.drop(columns=["cand_cc", "cosine_similarity"])

    temp_df = MCQS_data_pd[["property_id", "joyplot_label"]]
    wide = wide.merge(temp_df, left_on="target_id", right_on="property_id", how="inner")

    # ---- Choose colors (consistent per cand_cc) ----
    if got_groups:
        n_small = len(smallest_ccs)
        n_large = len(largest_ccs)

        small_colors = plt.cm.viridis(np.linspace(0.0, 0.5, n_small))
        large_colors = plt.cm.inferno(np.linspace(0.5, 0.9, n_large))

        colors = []
        for cc in ccs:
            if cc in smallest_ccs:
                colors.append(small_colors[smallest_ccs.index(cc)])
            elif cc in largest_ccs:
                colors.append(large_colors[largest_ccs.index(cc)])
            else:
                colors.append("grey")
    else:
        colors = [plt.cm.tab10(i % 10) for i in range(len(ccs))]

    # ---- Plot WITHOUT joypy's legend ----
    fig, axes = joypy.joyplot(
        wide,
        by="joyplot_label",
        column=ccs,
        color=colors,
        legend=False,  # IMPORTANT: we draw our own grouped legend
        overlap=1.5,
        linewidth=1,
        alpha=0.5,
        figsize=(8.5, 6.5),
        fade=False,
        ylim="own",
    )

    # axes is a list-like of Axes (one per ridge)
    ax0 = axes[0] if isinstance(axes, (list, np.ndarray)) else axes

    # ---- Axis labels ----
    # X label goes on the bottom axis (the only one with x ticks)
    if isinstance(axes, (list, np.ndarray)) and len(axes) > 0:
        axes[-1].set_xlabel("Cosine similarity", fontsize=11)

    # Y label is figure-level (since ridgeline has multiple axes)
    fig.text(
        0.02, 0.5,
        "Target property",
        va="center",
        rotation="vertical",
        fontsize=11
    )

    # ---- Single legend with two section "headers" ----
    if got_groups:
        cc_to_color = {cc: colors[i] for i, cc in enumerate(ccs)}

        handles = []

        # Section header: Small
        handles.append(mlines.Line2D([], [], color="none", label="Small countries"))
        for cc in smallest_ccs:
            handles.append(mpatches.Patch(color=cc_to_color[cc], label=f"  {cc}"))

        # Section header: Large
        handles.append(mlines.Line2D([], [], color="none", label="Large countries"))
        for cc in largest_ccs:
            handles.append(mpatches.Patch(color=cc_to_color[cc], label=f"  {cc}"))

        leg = ax0.legend(
            handles=handles,
            loc="upper right",  # inside plot
            frameon=True,
            fontsize=8,
        )

        # Make the section headers bold
        for t in leg.get_texts():
            if t.get_text().strip() in {"Small countries", "Large countries"}:
                t.set_weight("bold")

        # Hide marker for header lines
        for h, txt in zip(leg.legend_handles, leg.get_texts()):
            if txt.get_text().strip() in {"Small countries", "Large countries"}:
                try:
                    h.set_visible(False)
                except Exception:
                    pass

    else:
        # If no grouping provided, fall back to a normal flat legend (inside plot)
        cc_to_color = {cc: colors[i] for i, cc in enumerate(ccs)}
        handles = [mpatches.Patch(color=cc_to_color[cc], label=cc) for cc in ccs]
        ax0.legend(handles=handles, loc="upper right", frameon=True, fontsize=10)

    # ---- Title + save ----
    fig.suptitle(
        "Cosine Similarity by Destination Country (overlaid per property)",
        y=1.02,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_median_sum_table(
    suite_results_pd: pd.DataFrame,
    smallest_ccs=None,
    largest_ccs=None,
    out_path: str = "median_sum.csv",
):
    """
    Creates a table summarizing per-query median cosine similarities.
    Rows: target properties. Columns: destination countries.
    Adds:
      - last column: median across countries per target ("Target median")
      - last row: median across targets per country ("Country median")
      - bottom-right: global median of all query medians
    """
    smallest_ccs = smallest_ccs or []
    largest_ccs = largest_ccs or []
    ccs = list(smallest_ccs) + list(largest_ccs)

    # keep only the needed cols
    temp = suite_results_pd[["target_id", "cand_cc", "cosine_similarity"]].copy()

    # per-(target,country) median
    med = temp.groupby(["target_id", "cand_cc"], as_index=False)["cosine_similarity"].median()

    pivot = med.pivot(index="target_id", columns="cand_cc", values="cosine_similarity")

    # enforce desired column order
    pivot = pivot.reindex(columns=ccs)

    # add labels (safe merge)
    labels = MCQS_data_pd[["property_id", "listing_title"]]
    sum_table = pivot.reset_index().merge(
        labels, left_on="target_id", right_on="property_id", how="left"
    ).drop(columns=["property_id"])

    # put title first
    sum_table = sum_table[["target_id", "listing_title"] + ccs]

    # add per-target median across countries
    sum_table["target_property_median"] = sum_table[ccs].median(axis=1)
    
    # order the properties by target median
    sum_table = sum_table.sort_values(by="target_property_median", ascending=False)

    # add per-country median across targets + global median
    country_medians = sum_table[ccs].median(axis=0)
    global_median = sum_table[ccs].stack().median()

    last_row = {"target_id": "country_median", "listing_title": "Country Median"}
    last_row.update(country_medians.to_dict())
    last_row["target_property_median"] = global_median

    sum_table = pd.concat([sum_table, pd.DataFrame([last_row])], ignore_index=True)

    # round for readability
    for col in ccs + ["target_property_median"]:
        sum_table[col] = sum_table[col].round(3)

    sum_table.to_csv(out_path, index=False)
    return sum_table


# ---- Run ----
suite_results = spark.read.parquet(MCQS_RESULTS_PATH)
suite_results_pd = suite_results.toPandas()

create_ridge_plot(
    suite_results_pd,
    smallest_ccs=["GI", "JE", "AX"],
    largest_ccs=["FR", "IT", "ES"],
    out_path="ridgeline.png",
)

create_median_sum_table(
    suite_results_pd,
    smallest_ccs=["GI", "JE", "AX"],
    largest_ccs=["FR", "IT", "ES"],
    out_path="median_sum.csv",
)