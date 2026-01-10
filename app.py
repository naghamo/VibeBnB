import os
import json
from flask import Flask, render_template, request

app = Flask("vibebnb_app")

# ----------------------------
# Europe country codes (from your config; use only europe)
# ----------------------------
EUROPE_CC = set([
    "AD","AL","AM","AT","AX","AZ","BA","BE","BG","BY","CH","CY","CZ","DE","DK","DZ","EE","ES",
    "FI","FO","FR","GB","GE","GG","GI","GR","HR","HU","IE","IM","IQ","IR","IS","IT","JE","LI",
    "LT","LU","LV","MC","MD","ME","MK","MT","NL","NO","PL","PT","RO","RS","RU","SE","SI","SJ",
    "SK","SM","SY","TN","TR","UA","VA","XK"
])

# ----------------------------
# Static paths
# ----------------------------
UI_SAMPLE_PATH = os.environ.get(
    "VIBEBNB_UI_SAMPLE",
    os.path.join(app.root_path, "static", "listings_sample.json")
)
RESULTS_DIR = os.environ.get(
    "VIBEBNB_RESULTS_DIR",
    os.path.join(app.root_path, "static", "results")
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
BUDGET_CHOICES = ["Budget", "Mid-range", "Luxury"]

LISTINGS_UI = []
COUNTRIES = []

def _safe_upper(x: str | None) -> str:
    return str(x).upper().strip() if x is not None else ""

def load_ui_sample_europe_only():
    """
    Load sample listings from static JSON and filter to Europe only.
    Build countries dropdown from countries present in the sample (Europe-only).
    """
    global LISTINGS_UI, COUNTRIES

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

            cleaned.append(row)
            found_cc.add(cc)

        LISTINGS_UI = cleaned
        COUNTRIES = sorted(found_cc) if found_cc else sorted(EUROPE_CC)

        print(f"[APP] Loaded UI sample rows={len(LISTINGS_UI)} europe_countries={len(COUNTRIES)}")
        print(f"[APP] UI sample path: {UI_SAMPLE_PATH}")

    except Exception as e:
        print(f"[APP] Could not load UI sample: {e}")
        LISTINGS_UI = []
        COUNTRIES = sorted(EUROPE_CC)

load_ui_sample_europe_only()

def _result_filename(target_id: str, target_country: str) -> str:
    # Stable filename convention
    safe_id = str(target_id).strip()
    safe_cc = _safe_upper(target_country)
    return f"results_{safe_id}_{safe_cc}.json"

def load_precomputed_results(target_id: str, target_country: str):
    """
    Load results from static/results/results_{target_id}_{target_country}.json
    Expected JSON format:
      { "ok": true, "target": [...], "results": [...] }
    OR even just: { "results": [...] } (target optional)
    """
    fname = _result_filename(target_id, target_country)
    path = os.path.join(RESULTS_DIR, fname)

    if not os.path.exists(path):
        return None, path

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Normalize to expected structure
    results = payload.get("results", payload if isinstance(payload, list) else [])
    target_list = payload.get("target", [])
    target_row = target_list[0] if isinstance(target_list, list) and len(target_list) > 0 else None

    if isinstance(results, dict):
        # weird edge-case: results accidentally dict
        results = [results]

    return {"target_row": target_row, "results_rows": results, "payload": payload}, path

@app.route("/", methods=["GET", "POST"])
def index():
    result_msg = None
    target_row = None
    results_rows = None
    submitted = None

    if request.method == "POST":
        target_id = (request.form.get("selected_listing_id", "") or "").strip()
        target_country = _safe_upper(request.form.get("target_country", ""))

        submitted = {
            "target_id": target_id,
            "target_country": target_country
        }

        # Show selected target row from the sample table
        if target_id:
            for r in LISTINGS_UI:
                if r.get("property_id") == target_id:
                    target_row = r
                    break

        # Validate
        if not target_id:
            result_msg = "Please select a reference listing first."
        elif not target_country:
            result_msg = "Please choose a target country."
        elif target_country not in EUROPE_CC:
            result_msg = "Target country must be in Europe."
        else:
            # Try load precomputed results
            loaded, expected_path = load_precomputed_results(target_id, target_country)

            if loaded is None:
                # Not found: tell user exactly what file is missing
                result_msg = (
                    "No precomputed results found for this selection.\n\n"
                    f"Expected file:\n{expected_path}\n\n"
                    "Run the Spark job offline (Notebook/Workflow) to generate this JSON "
                    "and place it under static/results/."
                )
            else:
                # Merge: prefer target_row from file if present
                if loaded["target_row"] is not None:
                    target_row = loaded["target_row"]
                results_rows = loaded["results_rows"]

                result_msg = (
                    f"Loaded {len(results_rows)} precomputed results from:\n{expected_path}"
                )

    return render_template(
        "index.html",
        listings=LISTINGS_UI,
        countries=COUNTRIES,
        env_choices=ENV_CHOICES,
        budget_choices=BUDGET_CHOICES,
        result=result_msg,
        submitted=submitted,
        target_row=target_row,
        results_rows=results_rows
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
