# app.py
import os
import time
import json
import glob
import requests
from pathlib import Path
from flask import Flask, render_template, request

app = Flask("vibebnb_app")

# ----------------------------
# Europe country codes (for FILTERING)
# ----------------------------
EUROPE_CC = {
    "AD","AL","AM","AT","AX","AZ","BA","BE","BG","BY","CH","CY","CZ","DE","DK","DZ","EE","ES",
    "FI","FO","FR","GB","GE","GG","GI","GR","HR","HU","IE","IM","IQ","IR","IS","IT","JE","LI",
    "LT","LU","LV","MC","MD","ME","MK","MT","NL","NO","PL","PT","RO","RS","RU","SE","SI","SJ",
    "SK","SM","SY","TN","TR","UA","VA","XK"
}

# ----------------------------
# Target countries (ONLY these 5)
# ----------------------------
TARGET_COUNTRIES = ["FR", "IT", "ES", "GB", "DE"]

# ----------------------------
# Mode selection (OFFLINE ONLY)
# ----------------------------
VIBEBNB_MODE = "offline"

# Online placeholders 
JOB_ID = 0
WAREHOUSE_ID = ""
RESULTS_TABLE = os.environ.get("VIBEBNB_RESULTS_TABLE", "app_demo.vibebnb_results")
ONLINE_FILTER_LIMIT = int(os.environ.get("VIBEBNB_FILTER_LIMIT", "300") or "300")

# ----------------------------
# Offline config (UI sample + neighbors JSON)
# ----------------------------
UI_SAMPLE_PATH = os.environ.get(
    "VIBEBNB_UI_SAMPLE",
    os.path.join(app.root_path, "static", "listings_sample.json")
)

NEIGHBORS_JSON_DIR = os.environ.get(
    "VIBEBNB_NEIGHBORS_JSON_DIR",
    os.path.join(app.root_path, "static", "neighbors_json")
)

# ----------------------------
# UI config
# ----------------------------
UI_COLS = [
    "property_id", "addr_cc", "listing_title", "room_type_text",
    "addr_name", "price_per_night", "ratings"
]

ENV_CHOICES = [
    "env_food_norm", "env_nature_norm", "env_nightlife_norm", "env_transport_norm",
    "env_shopping_norm", "env_culture_norm", "env_leisure_norm", "env_services_norm"
]
BUDGET_CHOICES = ["", "Budget", "Mid-range", "Luxury"]

# fixed candidates count
N_CANDIDATES_FIXED = 50

OFFLINE_LISTINGS_ALL: list[dict] = []
COUNTRIES: list[str] = []


SESSION = requests.Session()

def _safe_upper(x) -> str:
    return str(x).upper().strip() if x is not None else ""

def load_ui_sample_europe_only():
    global OFFLINE_LISTINGS_ALL, COUNTRIES
    try:
        with open(UI_SAMPLE_PATH, "r", encoding="utf-8") as f:
            rows = json.load(f)

        cleaned = []
        found_cc = set()

        for r in rows:
            cc = _safe_upper(r.get("addr_cc"))
            if not cc or cc not in EUROPE_CC:
                continue

            pid = r.get("property_id")
            if pid is None:
                continue

            row = {c: r.get(c) for c in UI_COLS}
            row["property_id"] = str(pid)
            row["addr_cc"] = cc

            try:
                row["price_per_night"] = float(row.get("price_per_night")) if row.get("price_per_night") is not None else None
            except Exception:
                row["price_per_night"] = None

            try:
                row["ratings"] = float(row.get("ratings")) if row.get("ratings") is not None else None
            except Exception:
                row["ratings"] = None

            cleaned.append(row)
            found_cc.add(cc)

        OFFLINE_LISTINGS_ALL = cleaned
        COUNTRIES = sorted(found_cc) if found_cc else sorted(EUROPE_CC)

    except Exception as e:
        print(f"[APP] Could not load UI sample: {e}")
        OFFLINE_LISTINGS_ALL = []
        COUNTRIES = sorted(EUROPE_CC)

load_ui_sample_europe_only()

# ----------------------------
# Parsing helpers
# ----------------------------
def _to_int(x, default):
    try:
        return int(x)
    except Exception:
        return default

def _to_float(x, default):
    try:
        return float(x)
    except Exception:
        return default

def _normalize_weights_nonneg(weights: dict[str, float]) -> dict[str, float]:
    w = {k: max(0.0, float(v)) for k, v in (weights or {}).items()}
    s = sum(w.values())
    if s <= 0:
        return w
    return {k: v / s for k, v in w.items()}

# ----------------------------
# Filtering logic (OFFLINE)
# ----------------------------
def offline_filter_listings(country, city, min_rating, max_rating, min_price, max_price, limit=300):
    cc = _safe_upper(country) if country else ""
    city = (city or "").strip()

    rows = OFFLINE_LISTINGS_ALL
    if cc:
        rows = [r for r in rows if _safe_upper(r.get("addr_cc")) == cc]

    cities = sorted({(r.get("addr_name") or "").strip() for r in rows if (r.get("addr_name") or "").strip()})

    if city:
        rows = [r for r in rows if (r.get("addr_name") or "").strip() == city]

    def in_range(val, lo, hi):
        if val is None:
            return False
        if lo is not None and val < lo:
            return False
        if hi is not None and val > hi:
            return False
        return True

    if min_rating is not None or max_rating is not None:
        rows = [r for r in rows if in_range(r.get("ratings"), min_rating, max_rating)]

    if min_price is not None or max_price is not None:
        rows = [r for r in rows if in_range(r.get("price_per_night"), min_price, max_price)]

    rows = rows[:max(1, int(limit))]
    return rows, cities

# ----------------------------
# OFFLINE: load precomputed neighbors from JSON files
# Expected structure:
#   static/neighbors_json/target_cc=IT/<target_id>.json
#
# File format:
#   {"target_id": "...", "target_cc": "IT", "results": [ {...}, {...}, ... ] }
# ----------------------------
def offline_load_candidates(target_id: str, target_country: str):
    cc = _safe_upper(target_country)
    tid = str(target_id).strip()

    base_dir = os.path.join(NEIGHBORS_JSON_DIR, f"target_cc={cc}")
    path_used = os.path.join(base_dir, f"{tid}.json")

    # Support a "json/" subfolder if you used that variant
    alt_path = os.path.join(base_dir, "json", f"{tid}.json")

    if os.path.exists(path_used):
        chosen = path_used
    elif os.path.exists(alt_path):
        chosen = alt_path
    else:
        # Best-effort glob in case of slight naming differences
        found = glob.glob(os.path.join(base_dir, f"{tid}*.json")) + glob.glob(os.path.join(base_dir, "json", f"{tid}*.json"))
        chosen = found[0] if found else None

    if not chosen:
        return [], None

    with open(chosen, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = payload.get("results", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        rows = []
    return rows, chosen

# ----------------------------
# OFFLINE: local ranking
# ----------------------------
def offline_rank_candidates(
    candidates: list[dict],
    k_show: int,
    w_price: float,
    w_property: float,
    w_host: float,
    w_temp: float,
    w_budget: float,
    env_weights: dict[str, float],
    temp_pref: float | None,
    travel_month: int | None,
    budget_pref: str,
    normalize_all_weights: bool = True,
    score_col: str = "final_score",
):
    env_weights = env_weights or {}

    # --- normalize everything together (like Spark order(...)) ---
    all_w = {
        "price": w_price,
        "property": w_property,
        "host": w_host,
        "temp": w_temp,
        "budget": w_budget,
        **{f"env::{k}": v for k, v in env_weights.items()},
    }

    if normalize_all_weights:
        w_norm = _normalize_weights_nonneg(all_w)
    else:
        w_norm = {k: float(v) for k, v in all_w.items()}

    price_w = float(w_norm.get("price", 0.0))
    property_w = float(w_norm.get("property", 0.0))
    host_w = float(w_norm.get("host", 0.0))
    temp_w = float(w_norm.get("temp", 0.0))
    budget_w = float(w_norm.get("budget", 0.0))

    def getf(row, key, default=0.0):
        v = row.get(key, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    # --- Spark-like temperature score: 1 - diff/25 ---
    def temp_score_raw(row):
        if temp_pref is None or travel_month is None or not temp_w:
            return 0.0
        m = int(travel_month)
        if m < 1 or m > 12:
            return 0.0
        col = f"temp_avg_m{m:02d}"
        try:
            t = float(row.get(col))
        except Exception:
            return 0.0
        diff = abs(t - float(temp_pref))
        return max(0.0, 1.0 - diff / 25.0)

    # --- Spark-like budget score: partial credit by distance/2 ---
    def budget_score_raw(row):
        if not budget_w or not budget_pref:
            return 0.0

        rank_map = {"Budget": 1, "Mid-range": 2, "Luxury": 3}
        lvl = (row.get("budget_level") or "").strip()
        a = rank_map.get(lvl)
        b = rank_map.get((budget_pref or "").strip())
        if a is None or b is None:
            return 0.0
        return max(0.0, 1.0 - (abs(a - b) / 2.0))

    ranked = []
    for r in candidates:
        score = 0.0

        # base components
        if price_w:
            score += price_w * getf(r, "price_score", 0.0)
        if property_w:
            score += property_w * getf(r, "property_quality", 0.0)
        if host_w:
            score += host_w * getf(r, "host_quality", 0.0)

        # env components: expect *_norm columns (like Spark)
        for env_col, raw_w in env_weights.items():
            wv = float(w_norm.get(f"env::{env_col}", raw_w)) if normalize_all_weights else float(raw_w)
            if not wv:
                continue
            norm_col = env_col if str(env_col).endswith("_norm") else f"{env_col}_norm"
            score += wv * getf(r, norm_col, 0.0)

        # temp + budget
        if temp_w:
            score += temp_w * temp_score_raw(r)
        if budget_w:
            score += budget_w * budget_score_raw(r)

        rr = dict(r)
        rr[score_col] = float(score)
        ranked.append(rr)

    # Spark: orderBy(score desc).limit(k)
    # Tie-break: keep your l2_dist tie-breaker (fine; Spark version didn't specify tie-break)
    ranked.sort(key=lambda x: (-(x.get(score_col) or 0.0), x.get("l2_dist") or 1e9))
    return ranked[: max(1, int(k_show))]


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def info():
    return render_template("info.html", mode="offline", mode_reason="offline only")

@app.get("/filters")
def filters():
    """
    ONE PAGE:
      - Filters form
      - Listings table (after filtering)
    Filtering uses ALL Europe countries.
    """
    filter_country = _safe_upper(request.args.get("filter_country", "") or "")
    filter_city = (request.args.get("filter_city", "") or "").strip()

    min_rating = _to_float(request.args.get("min_rating", ""), None) if (request.args.get("min_rating", "") or "").strip() else None
    max_rating = _to_float(request.args.get("max_rating", ""), None) if (request.args.get("max_rating", "") or "").strip() else None
    min_price  = _to_float(request.args.get("min_price", ""), None) if (request.args.get("min_price", "") or "").strip() else None
    max_price  = _to_float(request.args.get("max_price", ""), None) if (request.args.get("max_price", "") or "").strip() else None

    listings = []
    city_choices = []
    status = None

    any_filter_used = any([
        filter_country, filter_city,
        min_rating is not None, max_rating is not None,
        min_price is not None, max_price is not None
    ])

    try:
        if any_filter_used:
            listings, city_choices = offline_filter_listings(
                filter_country, filter_city, min_rating, max_rating, min_price, max_price, ONLINE_FILTER_LIMIT
            )
            status = f"Found {len(listings)} listings (offline)."
        else:
            _, city_choices = offline_filter_listings(
                filter_country, "", None, None, None, None, ONLINE_FILTER_LIMIT
            )
            listings = []
            status = "Set filters and click Filter to see listings."
    except Exception as e:
        listings = []
        city_choices = []
        status = f"Filtering error (offline): {e}"

    countries = sorted(COUNTRIES) if COUNTRIES else sorted(EUROPE_CC)

    return render_template(
        "filters.html",
        mode="offline",
        mode_reason="offline only",
        countries=countries,
        city_choices=city_choices,
        listings=listings,
        filter_country=filter_country,
        filter_city=filter_city,
        min_rating=min_rating,
        max_rating=max_rating,
        min_price=min_price,
        max_price=max_price,
        status=status
    )

@app.route("/preferences", methods=["GET", "POST"])
def preferences():
    """
    Comes from filters page after selecting a reference listing.
    Target countries: ONLY 5 (FR, IT, ES, GB, DE).
    n_candidates: ALWAYS 50.
    """
    if request.method == "GET":
        return render_template("info.html", mode="offline", mode_reason="offline only")

    selected_listing_id = (request.form.get("selected_listing_id", "") or "").strip()

    # carry filters forward (for back link)
    filter_country = _safe_upper(request.form.get("filter_country", "") or "")
    filter_city = (request.form.get("filter_city", "") or "").strip()
    min_rating = _to_float(request.form.get("min_rating", ""), None) if (request.form.get("min_rating", "") or "").strip() else None
    max_rating = _to_float(request.form.get("max_rating", ""), None) if (request.form.get("max_rating", "") or "").strip() else None
    min_price  = _to_float(request.form.get("min_price", ""), None) if (request.form.get("min_price", "") or "").strip() else None
    max_price  = _to_float(request.form.get("max_price", ""), None) if (request.form.get("max_price", "") or "").strip() else None

    target_country = _safe_upper(request.form.get("target_country", "") or "")
    n_candidates = N_CANDIDATES_FIXED
    k_show = _to_int(request.form.get("k_show", "10"), 10)

    status = None
    if not selected_listing_id:
        status = "Please select a reference listing first."
        return render_template(
            "filters.html",
            mode="offline", mode_reason="offline only",
            countries=sorted(COUNTRIES) if COUNTRIES else sorted(EUROPE_CC),
            city_choices=[],
            listings=[],
            filter_country=filter_country, filter_city=filter_city,
            min_rating=min_rating, max_rating=max_rating,
            min_price=min_price, max_price=max_price,
            status=status
        )

    return render_template(
        "preferences.html",
        mode="offline",
        mode_reason="offline only",
        countries=TARGET_COUNTRIES,
        env_choices=ENV_CHOICES,
        budget_choices=BUDGET_CHOICES,

        selected_listing_id=selected_listing_id,

        filter_country=filter_country,
        filter_city=filter_city,
        min_rating=min_rating,
        max_rating=max_rating,
        min_price=min_price,
        max_price=max_price,

        target_country=target_country,
        n_candidates=n_candidates,
        k_show=k_show,

        status=status
    )

# @app.route("/results", methods=["GET", "POST"])
# def results():
#     if request.method == "GET":
#         return render_template("info.html", mode="offline", mode_reason="offline only")

#     target_id = (request.form.get("selected_listing_id", "") or "").strip()
#     target_country = _safe_upper(request.form.get("target_country", ""))

#     n_candidates = N_CANDIDATES_FIXED
#     k_show = _to_int(request.form.get("k_show", "10"), 10)

#     w_price = _to_float(request.form.get("w_price", "0"), 0.0) or 0.0
#     w_property = _to_float(request.form.get("w_property", "0"), 0.0) or 0.0
#     w_host = _to_float(request.form.get("w_host", "0"), 0.0) or 0.0
#     w_temp = _to_float(request.form.get("w_temp", "0"), 0.0) or 0.0
#     w_budget = _to_float(request.form.get("w_budget", "0"), 0.0) or 0.0

#     temp_pref_raw = (request.form.get("temp_pref", "") or "").strip()
#     temp_pref = _to_float(temp_pref_raw, None) if temp_pref_raw else None

#     travel_month_raw = (request.form.get("travel_month", "") or "").strip()
#     travel_month = _to_int(travel_month_raw, None) if travel_month_raw else None

#     budget_pref = (request.form.get("budget_pref", "") or "").strip() or ""

#     env_weights = {}
#     for c in ENV_CHOICES:
#         env_weights[c] = float(_to_float(request.form.get(f"w_{c}", "0"), 0.0) or 0.0)

#     results_rows = []
#     result_msg = None

#     if not target_id:
#         result_msg = "Missing selected_listing_id."
#         return render_template("results.html", mode="offline", mode_reason="offline only", results_rows=[], result=result_msg)

#     if not target_country:
#         result_msg = "Please choose a target country."
#         return render_template("results.html", mode="offline", mode_reason="offline only", results_rows=[], result=result_msg)

#     if target_country not in TARGET_COUNTRIES:
#         result_msg = f"Target country must be one of: {', '.join(TARGET_COUNTRIES)}."
#         return render_template("results.html", mode="offline", mode_reason="offline only", results_rows=[], result=result_msg)

#     try:
#         t0 = time.perf_counter()

#         candidates, path_used = offline_load_candidates(target_id, target_country)
#         if not candidates:
#             expected = Path(NEIGHBORS_JSON_DIR) / f"target_cc={target_country}" / f"{target_id}.json"
#             expected_alt = Path(NEIGHBORS_JSON_DIR) / f"target_cc={target_country}" / "json" / f"{target_id}.json"
#             raise RuntimeError(
#                 "Offline mode: no precomputed neighbors found.\n"
#                 f"Tried:\n- {expected}\n- {expected_alt}\n"
#                 f"Neighbors JSON dir: {NEIGHBORS_JSON_DIR}"
#             )

#         candidates = candidates[:N_CANDIDATES_FIXED]

#         results_rows = offline_rank_candidates(
#             candidates=candidates,
#             k_show=k_show,
#             w_price=w_price,
#             w_property=w_property,
#             w_host=w_host,
#             w_temp=w_temp,
#             w_budget=w_budget,
#             env_weights=env_weights,
#             temp_pref=temp_pref,
#             travel_month=travel_month,
#             budget_pref=budget_pref,
#         )

#         t1 = time.perf_counter()
#         result_msg = (
#             f"Offline: ranked top {len(results_rows)} "
#             f"(loaded {len(candidates)} / {N_CANDIDATES_FIXED} from {path_used}). "
#             f"({t1 - t0:.2f}s)"
#         )

#     except Exception as e:
#         result_msg = f"Error (offline): {e}"
#         results_rows = []

#     return render_template(
#         "results.html",
#         mode="offline",
#         mode_reason="offline only",
#         results_rows=results_rows,
#         result=result_msg
#     )
@app.route("/results", methods=["GET", "POST"])
def results():
    if request.method == "GET":
        return render_template("info.html", mode="offline", mode_reason="offline only")

    target_id = (request.form.get("selected_listing_id", "") or "").strip()
    target_country = _safe_upper(request.form.get("target_country", ""))

    # fixed
    n_candidates = N_CANDIDATES_FIXED
    k_show = _to_int(request.form.get("k_show", "10"), 10)

    w_price = _to_float(request.form.get("w_price", "0"), 0.0) or 0.0
    w_property = _to_float(request.form.get("w_property", "0"), 0.0) or 0.0
    w_host = _to_float(request.form.get("w_host", "0"), 0.0) or 0.0
    w_temp = _to_float(request.form.get("w_temp", "0"), 0.0) or 0.0
    w_budget = _to_float(request.form.get("w_budget", "0"), 0.0) or 0.0

    temp_pref_raw = (request.form.get("temp_pref", "") or "").strip()
    temp_pref = _to_float(temp_pref_raw, None) if temp_pref_raw else None

    travel_month_raw = (request.form.get("travel_month", "") or "").strip()
    travel_month = _to_int(travel_month_raw, None) if travel_month_raw else None

    budget_pref = (request.form.get("budget_pref", "") or "").strip() or ""

    env_weights = {}
    for c in ENV_CHOICES:
        env_weights[c] = float(_to_float(request.form.get(f"w_{c}", "0"), 0.0) or 0.0)

    if not target_id:
        return render_template(
            "results.html",
            mode="offline",
            mode_reason="offline only",
            results_rows=[],
            result="Missing selected_listing_id."
        )

    if not target_country:
        return render_template(
            "results.html",
            mode="offline",
            mode_reason="offline only",
            results_rows=[],
            result="Please choose a target country."
        )

    if target_country not in TARGET_COUNTRIES:
        return render_template(
            "results.html",
            mode="offline",
            mode_reason="offline only",
            results_rows=[],
            result=f"Target country must be one of: {', '.join(TARGET_COUNTRIES)}."
        )

    try:
        t0 = time.perf_counter()

        # Load precomputed neighbors
        candidates, path_used = offline_load_candidates(target_id, target_country)
        if not candidates:
            expected_folder = Path(NEIGHBORS_JSON_DIR) / f"target_cc={target_country}"
            raise RuntimeError(
                "Offline mode: no precomputed neighbors found.\n"
                f"Expected folder like: {expected_folder}\n"
                "Make sure the folder exists under static/neighbors and contains the JSON files."
            )

        candidates = candidates[:n_candidates]

        ranked = offline_rank_candidates(
            candidates=candidates,
            k_show=k_show,
            w_price=w_price,
            w_property=w_property,
            w_host=w_host,
            w_temp=w_temp,
            w_budget=w_budget,
            env_weights=env_weights,
            temp_pref=temp_pref,
            travel_month=travel_month,
            budget_pref=budget_pref,
        )

        # ✅ UI: keep only user-facing fields (hide dist/similarity)
        KEEP = [
            "listing_title",
            "addr_cc",
            "addr_name",
            "room_type_text",
            "price_per_night",
            "ratings",
            "final_url",
        ]
        results_rows = [{k: r.get(k) for k in KEEP} for r in ranked]

        t1 = time.perf_counter()
        result_msg = (
            f"Offline: ranked top {len(results_rows)} "
            f"(loaded {len(candidates)} / {N_CANDIDATES_FIXED} from {path_used}). "
            f"({t1 - t0:.2f}s)"
        )

    except Exception as e:
        result_msg = f"Error (offline): {e}"
        results_rows = []

    return render_template(
        "results.html",
        mode="offline",
        mode_reason="offline only",
        results_rows=results_rows,
        result=result_msg
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8800"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
