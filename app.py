


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
# Europe country codes
# ----------------------------
EUROPE_CC = {
    "AD","AL","AM","AT","AX","AZ","BA","BE","BG","BY","CH","CY","CZ","DE","DK","DZ","EE","ES",
    "FI","FO","FR","GB","GE","GG","GI","GR","HR","HU","IE","IM","IQ","IR","IS","IT","JE","LI",
    "LT","LU","LV","MC","MD","ME","MK","MT","NL","NO","PL","PT","RO","RS","RU","SE","SI","SJ",
    "SK","SM","SY","TN","TR","UA","VA","XK"
}

# ----------------------------
# Mode selection
# ----------------------------
VIBEBNB_MODE = (os.environ.get("VIBEBNB_MODE", "auto") or "auto").strip().lower()

# ----------------------------
# Online config (Job + SQL Warehouse)
# ----------------------------
JOB_ID = int(os.environ.get("VIBEBNB_JOB_ID", "0") or "0")
WAREHOUSE_ID = os.environ.get("DATABRICKS_SQL_ID", "") or ""
RESULTS_TABLE = os.environ.get("VIBEBNB_RESULTS_TABLE", "app_demo.vibebnb_results")
ONLINE_FILTER_LIMIT = int(os.environ.get("VIBEBNB_FILTER_LIMIT", "300") or "300")

# ----------------------------
# Offline config (UI sample + neighbors)
# ----------------------------
UI_SAMPLE_PATH = os.environ.get(
    "VIBEBNB_UI_SAMPLE",
    os.path.join(app.root_path, "static", "listings_sample.json")
)
NEIGHBORS_DIR = os.environ.get(
    "VIBEBNB_NEIGHBORS_DIR",
    os.path.join(app.root_path, "static", "neighbors")
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

OFFLINE_LISTINGS_ALL: list[dict] = []
COUNTRIES: list[str] = []

# ----------------------------
# Databricks auth helpers (ONLINE)
# ----------------------------
SESSION = requests.Session()

def _safe_upper(x) -> str:
    return str(x).upper().strip() if x is not None else ""

def _get_databricks_host() -> str | None:
    for k in ["DATABRICKS_HOST", "DATABRICKS_WORKSPACE_URL", "WORKSPACE_URL", "DATABRICKS_URL"]:
        v = os.environ.get(k)
        if v:
            if not v.startswith("http"):
                v = "https://" + v
            return v.rstrip("/")
    return None

def _get_pat_token() -> str | None:
    for k in ["DATABRICKS_TOKEN", "DATABRICKS_AUTH_TOKEN", "DBX_TOKEN", "ACCESS_TOKEN"]:
        v = os.environ.get(k)
        if v:
            return v
    return None

def _get_oauth_client() -> tuple[str | None, str | None]:
    return os.environ.get("DATABRICKS_CLIENT_ID"), os.environ.get("DATABRICKS_CLIENT_SECRET")

def _url(host: str, path: str) -> str:
    return f"{host}{path}"

def _mint_oauth_token(host: str, client_id: str, client_secret: str, timeout_s: int = 20) -> str:
    token_url = _url(host, "/oidc/v1/token")
    data = {
        "grant_type": "client_credentials",
        "scope": "all-apis",
        "client_id": client_id,
        "client_secret": client_secret,
    }
    r = requests.post(token_url, data=data, timeout=timeout_s)
    r.raise_for_status()
    tok = (r.json() or {}).get("access_token")
    if not tok:
        raise RuntimeError("OAuth token response missing access_token.")
    return tok

def _ensure_databricks_session() -> tuple[str, str]:
    host = _get_databricks_host()
    if not host:
        raise RuntimeError("Missing DATABRICKS_HOST (workspace URL).")

    token = _get_pat_token()
    if token:
        SESSION.headers.update({"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        return host, token

    client_id, client_secret = _get_oauth_client()
    if client_id and client_secret:
        token = _mint_oauth_token(host, client_id, client_secret)
        SESSION.headers.update({"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        return host, token

    raise RuntimeError(
        "No Databricks credentials found. Provide either:\n"
        "- DATABRICKS_TOKEN, OR\n"
        "- DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET."
    )

# ----------------------------
# Load OFFLINE sample for filtering
# ----------------------------
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
# OFFLINE: neighbors + local ranking
# ----------------------------
def offline_load_candidates(target_id: str, target_country: str):
    cc = _safe_upper(target_country)
    tid = str(target_id).strip()

    patterns = [
        os.path.join(NEIGHBORS_DIR, cc, f"{tid}.json"),
        os.path.join(NEIGHBORS_DIR, f"target_cc={cc}", f"{tid}.json"),
        os.path.join(NEIGHBORS_DIR, cc, f"{tid}*.json"),
        os.path.join(NEIGHBORS_DIR, f"target_cc={cc}", f"{tid}*.json"),
    ]

    path_used = None
    for pat in patterns:
        found = glob.glob(pat)
        if found:
            path_used = found[0]
            break

    if not path_used:
        return [], None

    with open(path_used, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = payload.get("results", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        rows = []
    return rows, path_used

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
    budget_pref: str
):
    main = _normalize_weights_nonneg({
        "price": w_price,
        "property": w_property,
        "host": w_host,
        "temp": w_temp,
        "budget": w_budget,
    })
    env_w = _normalize_weights_nonneg(env_weights or {})

    def getf(row, key, default=0.0):
        v = row.get(key, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    def budget_match_score(row):
        if not budget_pref:
            return 0.0
        lvl = (row.get("budget_level") or "").strip()
        return 1.0 if lvl.lower() == budget_pref.lower() else 0.0

    def temp_match_score(row):
        if temp_pref is None or travel_month is None:
            return 0.0
        m = int(travel_month)
        if m < 1 or m > 12:
            return 0.0
        col = f"temp_avg_m{m:02d}"
        t = row.get(col, None)
        try:
            t = float(t)
        except Exception:
            return 0.0
        diff = abs(t - float(temp_pref))
        return max(0.0, 1.0 - diff / 15.0)

    ranked = []
    for r in candidates:
        price_s = getf(r, "price_score", 0.0)
        prop_s  = getf(r, "property_quality", 0.0)
        host_s  = getf(r, "host_quality", 0.0)

        env_s = 0.0
        for col, w in env_w.items():
            env_s += w * getf(r, col, 0.0)

        temp_s = temp_match_score(r)
        budg_s = budget_match_score(r)

        final = (
            main["price"] * price_s +
            main["property"] * prop_s +
            main["host"] * host_s +
            env_s +
            main["temp"] * temp_s +
            main["budget"] * budg_s
        )

        rr = dict(r)
        rr["final_score"] = float(final)
        ranked.append(rr)

    ranked.sort(key=lambda x: (-(x.get("final_score") or 0.0), x.get("l2_dist") or 1e9))
    return ranked[: max(1, int(k_show))]

# ----------------------------
# ONLINE: Job runner (supports 2 actions)
# ----------------------------
def run_job_and_get_result_string(notebook_params: dict, timeout_s: int = 300, poll_s: float = 1.5):
    if JOB_ID <= 0:
        raise RuntimeError("VIBEBNB_JOB_ID is missing/invalid.")

    host, _ = _ensure_databricks_session()

    resp = SESSION.post(
        _url(host, "/api/2.1/jobs/run-now"),
        data=json.dumps({"job_id": JOB_ID, "notebook_params": notebook_params}),
        timeout=30
    )
    resp.raise_for_status()
    run_id = resp.json()["run_id"]

    t0 = time.time()
    while True:
        r = SESSION.get(_url(host, "/api/2.1/jobs/runs/get"), params={"run_id": run_id}, timeout=30)
        r.raise_for_status()
        info = r.json()

        state = info.get("state", {})
        life = state.get("life_cycle_state")
        result = state.get("result_state")
        msg = state.get("state_message", "")

        if life == "TERMINATED":
            if result != "SUCCESS":
                raise RuntimeError(f"Job failed: result_state={result}. {msg}")
            break

        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Timed out after {timeout_s}s (run_id={run_id})")

        time.sleep(poll_s)

    out = SESSION.get(_url(host, "/api/2.1/jobs/runs/get-output"), params={"run_id": run_id}, timeout=30)
    out.raise_for_status()
    out_json = out.json()

    result_str = out_json.get("notebook_output", {}).get("result")
    if not result_str:
        raise RuntimeError("No notebook_output.result found. Notebook must exit via dbutils.notebook.exit(...).")

    return str(result_str), int(run_id)

def sql_fetch_results_by_request_id(request_id: str, limit: int = 200):
    if not WAREHOUSE_ID:
        raise RuntimeError("Missing DATABRICKS_SQL_ID (SQL Warehouse ID).")

    host, _ = _ensure_databricks_session()

    statement = f"""
    SELECT
      request_id, created_at, target_id, target_country, rank,
      property_id, addr_cc, listing_title, addr_name, room_type_text,
      price_per_night, ratings, l2_dist, final_score, final_url
    FROM {RESULTS_TABLE}
    WHERE request_id = '{request_id}'
    ORDER BY rank
    LIMIT {int(limit)}
    """

    payload = {
        "warehouse_id": WAREHOUSE_ID,
        "statement": statement,
        "disposition": "INLINE",
        "wait_timeout": "10s",
        "format": "JSON_ARRAY",
    }

    r = SESSION.post(_url(host, "/api/2.0/sql/statements/"), json=payload, timeout=30)
    r.raise_for_status()
    j = r.json()
    statement_id = j.get("statement_id")
    if not statement_id:
        raise RuntimeError("SQL API response missing statement_id.")

    t0 = time.time()
    while True:
        g = SESSION.get(_url(host, f"/api/2.0/sql/statements/{statement_id}"), timeout=30)
        g.raise_for_status()
        sj = g.json()

        status = (sj.get("status") or {}).get("state")
        if status == "SUCCEEDED":
            break
        if status in ("FAILED", "CANCELED", "CLOSED"):
            err = (sj.get("status") or {}).get("error") or {}
            raise RuntimeError(f"SQL statement {status}: {err}")

        if time.time() - t0 > 60:
            raise TimeoutError("SQL query timed out (>60s).")

        time.sleep(0.8)

    result = sj.get("result") or {}
    data = result.get("data_array") or []
    schema = (sj.get("manifest") or {}).get("schema") or {}
    cols = [c.get("name") for c in (schema.get("columns") or [])]

    rows = []
    for arr in data:
        row = {}
        for i, v in enumerate(arr):
            if i < len(cols):
                row[cols[i]] = v
        rows.append(row)
    return rows

# ----------------------------
# ONLINE filter via Job
# ----------------------------
def online_filter_listings_via_job(filter_country, filter_city, min_rating, max_rating, min_price, max_price, limit):
    params = {
        "action": "filter_listings",
        "filter_country": _safe_upper(filter_country),
        "filter_city": (filter_city or "").strip(),
        "min_rating": "" if min_rating is None else str(min_rating),
        "max_rating": "" if max_rating is None else str(max_rating),
        "min_price": "" if min_price is None else str(min_price),
        "max_price": "" if max_price is None else str(max_price),
        "limit": str(int(limit)),
    }
    result_str, _ = run_job_and_get_result_string(params)
    try:
        payload = json.loads(result_str)
    except Exception:
        raise RuntimeError("Filter job returned non-JSON output.")

    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "Unknown error in filter_listings job."))

    listings = payload.get("listings", [])
    cities = payload.get("cities", [])
    if not isinstance(listings, list): listings = []
    if not isinstance(cities, list): cities = []
    return listings, cities

# ----------------------------
# Mode resolver
# ----------------------------
def _online_ready():
    if JOB_ID <= 0:
        return False, "Missing/invalid VIBEBNB_JOB_ID"
    try:
        _ensure_databricks_session()
    except Exception as e:
        return False, f"Databricks auth not ready: {e}"
    return True, "OK"

def _choose_mode():
    if VIBEBNB_MODE == "offline":
        return "offline", "forced by VIBEBNB_MODE"
    if VIBEBNB_MODE == "online":
        ok, why = _online_ready()
        return ("online", "forced by VIBEBNB_MODE") if ok else ("offline", f"online forced but not ready: {why}")
    ok, why = _online_ready()
    if ok:
        return "online", "auto-selected (online ready)"
    return "offline", f"auto-selected (online not ready: {why})"

# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def info():
    mode, mode_reason = _choose_mode()
    return render_template("info.html", mode=mode, mode_reason=mode_reason)

@app.get("/filters")
def filters():
    """
    ONE PAGE:
      - Filters form
      - Listings table (after filtering)
      - User can change filters and click Filter again
    """
    mode, mode_reason = _choose_mode()

    # Read filter args (GET)
    filter_country = _safe_upper(request.args.get("filter_country", "") or "")
    filter_city = (request.args.get("filter_city", "") or "").strip()

    min_rating = _to_float(request.args.get("min_rating", ""), None) if (request.args.get("min_rating", "") or "").strip() else None
    max_rating = _to_float(request.args.get("max_rating", ""), None) if (request.args.get("max_rating", "") or "").strip() else None
    min_price  = _to_float(request.args.get("min_price", ""), None) if (request.args.get("min_price", "") or "").strip() else None
    max_price  = _to_float(request.args.get("max_price", ""), None) if (request.args.get("max_price", "") or "").strip() else None

    # Always compute city choices for selected country
    # If online: ask job (city list) only when a country is selected or any filter exists
    listings = []
    city_choices = []
    status = None

    any_filter_used = any([
        filter_country, filter_city,
        min_rating is not None, max_rating is not None,
        min_price is not None, max_price is not None
    ])

    try:
        if mode == "online":
            if any_filter_used:
                listings, city_choices = online_filter_listings_via_job(
                    filter_country, filter_city, min_rating, max_rating, min_price, max_price, ONLINE_FILTER_LIMIT
                )
                status = f"Found {len(listings)} listings (online)."
            else:
                # no filter yet -> still show country dropdown but empty city list + empty table
                listings, city_choices = [], []
                status = "Set filters and click Filter to see listings."
        else:
            # offline: we can always compute city choices even without pressing filter
            # but keep listings empty until user applies any filter (optional)
            if any_filter_used:
                listings, city_choices = offline_filter_listings(
                    filter_country, filter_city, min_rating, max_rating, min_price, max_price, ONLINE_FILTER_LIMIT
                )
                status = f"Found {len(listings)} listings (offline)."
            else:
                # compute city_choices based on country selection (if any)
                _, city_choices = offline_filter_listings(
                    filter_country, "", None, None, None, None, ONLINE_FILTER_LIMIT
                )
                listings = []
                status = "Set filters and click Filter to see listings."
    except Exception as e:
        listings = []
        city_choices = []
        status = f"Filtering error ({mode}): {e}"

    countries = sorted(COUNTRIES) if COUNTRIES else sorted(EUROPE_CC)

    return render_template(
        "filters.html",
        mode=mode,
        mode_reason=mode_reason,
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
    """
    if request.method == "GET":
        # If someone opens /preferences directly, redirect to filters
        return render_template("info.html", mode=_choose_mode()[0], mode_reason=_choose_mode()[1])

    mode, mode_reason = _choose_mode()

    selected_listing_id = (request.form.get("selected_listing_id", "") or "").strip()

    # carry filters forward (for back link)
    filter_country = _safe_upper(request.form.get("filter_country", "") or "")
    filter_city = (request.form.get("filter_city", "") or "").strip()
    min_rating = _to_float(request.form.get("min_rating", ""), None) if (request.form.get("min_rating", "") or "").strip() else None
    max_rating = _to_float(request.form.get("max_rating", ""), None) if (request.form.get("max_rating", "") or "").strip() else None
    min_price  = _to_float(request.form.get("min_price", ""), None) if (request.form.get("min_price", "") or "").strip() else None
    max_price  = _to_float(request.form.get("max_price", ""), None) if (request.form.get("max_price", "") or "").strip() else None

    # defaults for this screen
    target_country = _safe_upper(request.form.get("target_country", "") or "")
    n_candidates = _to_int(request.form.get("n_candidates", "50"), 50)
    k_show = _to_int(request.form.get("k_show", "10"), 10)

    status = None
    if not selected_listing_id:
        status = "Please select a reference listing first."
        # redirect back to filters with same args
        return render_template(
            "filters.html",
            mode=mode, mode_reason=mode_reason,
            countries=sorted(COUNTRIES) if COUNTRIES else sorted(EUROPE_CC),
            city_choices=[],
            listings=[],
            filter_country=filter_country, filter_city=filter_city,
            min_rating=min_rating, max_rating=max_rating,
            min_price=min_price, max_price=max_price,
            status=status
        )

    countries = sorted(COUNTRIES) if COUNTRIES else sorted(EUROPE_CC)

    return render_template(
        "preferences.html",
        mode=mode,
        mode_reason=mode_reason,
        countries=countries,
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

@app.route("/results", methods=["GET", "POST"])
def results():
    if request.method == "GET":
        # direct open -> go to filters
        return render_template("info.html", mode=_choose_mode()[0], mode_reason=_choose_mode()[1])

    mode, mode_reason = _choose_mode()

    target_id = (request.form.get("selected_listing_id", "") or "").strip()
    target_country = _safe_upper(request.form.get("target_country", ""))

    n_candidates = _to_int(request.form.get("n_candidates", "50"), 50)
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

    results_rows = []
    result_msg = None

    if not target_id:
        result_msg = "Missing selected_listing_id."
        return render_template("results.html", mode=mode, mode_reason=mode_reason, results_rows=[], result=result_msg)

    if not target_country:
        result_msg = "Please choose a target country."
        return render_template("results.html", mode=mode, mode_reason=mode_reason, results_rows=[], result=result_msg)

    if target_country not in EUROPE_CC:
        result_msg = "Target country must be in Europe."
        return render_template("results.html", mode=mode, mode_reason=mode_reason, results_rows=[], result=result_msg)

    try:
        t0 = time.perf_counter()

        if mode == "offline":
            candidates, path_used = offline_load_candidates(target_id, target_country)
            if not candidates:
                raise RuntimeError(
                    "Offline mode: no precomputed neighbors found.\n"
                    f"Expected file like: {Path(NEIGHBORS_DIR)/target_country/(target_id+'.json')}"
                )
            candidates = candidates[: max(1, int(n_candidates))]
            results_rows = offline_rank_candidates(
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
            result_msg = f"Offline: ranked top {len(results_rows)} (loaded {len(candidates)} from {path_used})."

        else:
            if not WAREHOUSE_ID:
                raise RuntimeError("Missing DATABRICKS_SQL_ID (SQL Warehouse ID) required for online results.")

            notebook_params = {
                "action": "recommend",
                "target_id": target_id,
                "target_country": target_country,
                "n_candidates": str(n_candidates),
                "k_show": str(k_show),
                "w_price": str(w_price),
                "w_property": str(w_property),
                "w_host": str(w_host),
                "w_temp": str(w_temp),
                "w_budget": str(w_budget),
                "temp_pref": "" if temp_pref is None else str(temp_pref),
                "travel_month": "" if travel_month is None else str(travel_month),
                "budget_pref": str(budget_pref),
                "env_weights_json": json.dumps(env_weights),
                "results_table": RESULTS_TABLE,
                "cleanup_old": "1",
            }

            result_str, run_id = run_job_and_get_result_string(notebook_params)
            request_id = result_str.strip()
            results_rows = sql_fetch_results_by_request_id(request_id, limit=max(50, k_show))
            result_msg = f"Online: run {run_id} finished | request_id={request_id} | rows={len(results_rows)}."

        t1 = time.perf_counter()
        result_msg = (result_msg or "") + f" (mode={mode}, {t1 - t0:.2f}s)"

    except Exception as e:
        result_msg = f"Error ({mode}): {e}"
        results_rows = []

    return render_template(
        "results.html",
        mode=mode,
        mode_reason=mode_reason,
        results_rows=results_rows,
        result=result_msg
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8800"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
