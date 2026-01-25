"""
OSM POI Extraction and Geocoding Pipeline

Extracts Points of Interest from OpenStreetMap PBF files and enriches them
with reverse geocoding information.
"""

import os
import sys
import argparse
import osmium
import pyarrow as pa
import pyarrow.parquet as pq
from shapely import from_wkb
from collections import defaultdict
from tqdm.auto import tqdm
import pandas as pd
import reverse_geocoder as rg
from pathlib import Path
import time


# ================= POI CONFIGURATION =================

POI_MAPPING = {
    # # --- ACCOMMODATION ---
    # ("tourism","hotel"): ("hotel","Accommodation"),
    # ("tourism","hostel"): ("hostel","Accommodation"),
    # ("tourism","guest_house"): ("guest_house","Accommodation"),
    # ("tourism","bed_and_breakfast"): ("b_and_b","Accommodation"),
    # ("tourism","apartment"): ("apartment","Accommodation"),
    # ("tourism","chalet"): ("chalet","Accommodation"),
    # ("tourism","camp_site"): ("camp_site","Accommodation"),
    # ("tourism","caravan_site"): ("caravan_site","Accommodation"),
    # ("tourism","alpine_hut"): ("alpine_hut","Accommodation"),

    # --- FOOD ---
    ("amenity","restaurant"): ("restaurant","Food"),
    ("amenity","fast_food"): ("fast_food","Food"),
    ("amenity","cafe"): ("cafe","Food"),
    ("amenity","food_court"): ("food_court","Food"),
    ("amenity","ice_cream"): ("ice_cream","Food"),
    ("amenity","biergarten"): ("biergarten","Food"),

    # --- NIGHTLIFE ---
    ("amenity","bar"): ("bar","Nightlife"),
    ("amenity","pub"): ("pub","Nightlife"),
    ("amenity","nightclub"): ("nightclub","Nightlife"),
    ("amenity","casino"): ("casino","Nightlife"),

    # --- CULTURE ---
    ("tourism","museum"): ("museum","Culture"),
    ("amenity","arts_centre"): ("arts_centre","Culture"),
    ("amenity","theatre"): ("theatre","Culture"),
    ("amenity","cinema"): ("cinema","Culture"),
    ("tourism","gallery"): ("gallery","Culture"),
    ("amenity","library"): ("library","Culture"),

    # --- SIGHTSEEING ---
    ("tourism","attraction"): ("attraction","Sightseeing"),
    ("tourism","viewpoint"): ("viewpoint","Sightseeing"),
    ("tourism","artwork"): ("artwork","Sightseeing"),
    ("historic","monument"): ("monument","Sightseeing"),
    ("historic","memorial"): ("memorial","Sightseeing"),
    ("historic","castle"): ("castle","Sightseeing"),
    ("historic","ruins"): ("ruins","Sightseeing"),
    ("historic","archaeological_site"): ("archaeological","Sightseeing"),
    ("historic","fort"): ("fort","Sightseeing"),
    ("amenity","place_of_worship"): ("place_of_worship","Sightseeing"),

    # --- NATURE ---
    ("natural","beach"): ("beach","Nature"),
    ("natural","peak"): ("peak","Nature"),
    ("natural","volcano"): ("volcano","Nature"),
    ("natural","cave_entrance"): ("cave_entrance","Nature"),
    ("natural","glacier"): ("glacier","Nature"),
    ("leisure","park"): ("park","Nature"),
    ("leisure","garden"): ("garden","Nature"),
    ("leisure","nature_reserve"): ("nature_reserve","Nature"),

    # --- LEISURE / FAMILY ---
    ("leisure","playground"): ("playground","Family"),
    ("tourism","zoo"): ("zoo","Family"),
    ("leisure","water_park"): ("water_park","Family"),
    ("leisure","swimming_pool"): ("swimming_pool","Family"),
    ("leisure","stadium"): ("stadium","Leisure"),
    ("leisure","sports_centre"): ("sports_centre","Leisure"),
    ("leisure","marina"): ("marina","Leisure"),
    ("tourism","picnic_site"): ("picnic_site","Leisure"),

    # --- SUPPLIES ---
    ("shop","supermarket"): ("supermarket","Supplies"),
    ("shop","convenience"): ("convenience","Supplies"),
    ("shop","bakery"): ("bakery","Supplies"),
    ("shop","greengrocer"): ("greengrocer","Supplies"),
    ("shop","general"): ("general_store","Supplies"),

    # --- SHOPPING ---
    ("shop","mall"): ("mall","Shopping"),
    ("shop","department_store"): ("department_store","Shopping"),
    ("shop","clothes"): ("clothes","Shopping"),
    ("shop","gift"): ("gift_shop","Shopping"),
    ("shop","books"): ("bookshop","Shopping"),

    # --- SERVICES ---
    ("tourism","information"): ("tourist_info","Services"),
    ("amenity","toilets"): ("toilets","Services"),
    ("amenity","bank"): ("bank","Services"),
    ("amenity","atm"): ("atm","Services"),
    ("amenity","post_office"): ("post_office","Services"),
    ("shop","laundry"): ("laundry","Services"),
    ("amenity","car_rental"): ("car_rental","Services"),
    ("amenity","bicycle_rental"): ("bicycle_rental","Services"),
    ("amenity","travel_agent"): ("travel_agent","Services"),

    # --- HEALTH & SAFETY ---
    ("amenity","pharmacy"): ("pharmacy","Health"),
    ("amenity","hospital"): ("hospital","Health"),
    ("amenity","clinic"): ("clinic","Health"),
    ("amenity","doctors"): ("doctors","Health"),
    ("amenity","police"): ("police","Health"),

    # --- TRANSPORT ---
    ("railway","station"): ("train_station","Transport"),
    # ("railway","tram_stop"): ("tram_stop","Transport"),
    ("railway","subway_entrance"): ("subway_entrance","Transport"),
    # ("highway","bus_stop"): ("bus_stop","Transport"),
    ("amenity","bus_station"): ("bus_station","Transport"),
    ("amenity","taxi"): ("taxi","Transport"),
    ("amenity","ferry_terminal"): ("ferry_terminal","Transport"),
    ("amenity","airport"): ("airport","Transport"),
}

GROUP_PRIORITY = {
    "Accommodation": 1,
    "Sightseeing": 2,
    "Culture": 3,
    "Family": 4,
    "Nightlife": 5,
    "Food": 6,
    "Nature": 7,
    "Transport": 8,
    "Leisure": 9,
    "Shopping": 10,
    "Supplies": 11,
    "Services": 12,
    "Health": 13
}


# ================= EXCEPTIONS =================

class LimitReached(Exception):
    """Raised when max_pois limit is reached"""
    pass


# ================= POI HANDLER =================

class POIHandler(osmium.SimpleHandler):
    """Handles OSM data and extracts POIs"""

    def __init__(self, output_path, total_elements=None, batch_size=500_000,
                 max_pois=None, step=1):
        super().__init__()
        self.wkb_factory = osmium.geom.WKBFactory()
        self.columns = defaultdict(list)
        self.output_path = output_path
        self.batch_size = batch_size
        self.writer = None
        self.max_pois = max_pois
        self.step = step

        # Counters
        self.total_pois = 0
        self.batch_count = 0
        self.batch_num = 0
        self.scanned_count = 0

        self.pbar = tqdm(total=total_elements, unit=" elems", desc="Extracting POIs",
                        mininterval=2.0)

        # Schema
        self.schema = pa.schema([
            ("osm_type", pa.string()),
            ("osm_id", pa.int64()),
            ("lat", pa.float64()),
            ("lon", pa.float64()),
            ("poi_class", pa.string()),
            ("poi_group", pa.string()),
            ("tags", pa.map_(pa.string(), pa.string())),
        ])

    def _extract_poi(self, tags, geom_func, osm_type, osm_id):
        """Extract POI from OSM element"""
        # Find all matches
        matches = []
        for k, v in tags:
            if (k, v) in POI_MAPPING:
                p_class, p_group = POI_MAPPING[(k, v)]
                priority = GROUP_PRIORITY.get(p_group, 99)
                matches.append((priority, p_class, p_group))

        if not matches:
            return

        # Pick best match (lowest priority number wins)
        matches.sort(key=lambda x: x[0])
        poi_class, poi_group = matches[0][1], matches[0][2]

        # Get geometry
        try:
            wkb_data = geom_func()
            geom = from_wkb(wkb_data)
            centroid = geom.centroid
        except Exception:
            return

        # Save all tags
        tag_items = [(t.k, t.v) for t in tags]

        self.columns["osm_type"].append(osm_type)
        self.columns["osm_id"].append(osm_id)
        self.columns["lat"].append(centroid.y)
        self.columns["lon"].append(centroid.x)
        self.columns["poi_class"].append(poi_class)
        self.columns["poi_group"].append(poi_group)
        self.columns["tags"].append(tag_items)

        self.batch_count += 1
        self.total_pois += 1

        if self.total_pois % 1000 == 0:
            self.pbar.set_postfix({"POIs": f"{self.total_pois:,}"})

        if self.max_pois and self.total_pois >= self.max_pois:
            raise LimitReached()

        if self.batch_count >= self.batch_size:
            self._flush()

    def _flush(self):
        """Write accumulated data to parquet"""
        if not self.columns["osm_id"]:
            return

        table = pa.Table.from_pydict(self.columns, schema=self.schema)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.output_path, self.schema,
                                          compression='snappy')
        self.writer.write_table(table)
        self.batch_num += 1

        # Cleanup
        del table
        self.columns = defaultdict(list)
        self.batch_count = 0

    def node(self, n):
        """Process OSM node"""
        self.scanned_count += 1
        if self.scanned_count % 50000 == 0:
            self.pbar.update(50000)

        if self.scanned_count % self.step != 0:
            return

        if not n.tags:
            return

        try:
            self._extract_poi(n.tags, lambda: self.wkb_factory.create_point(n),
                            "node", n.id)
        except osmium.InvalidLocationError:
            pass

    def area(self, a):
        """Process OSM area (way or relation)"""
        self.scanned_count += 1
        if self.scanned_count % 1000 == 0:
            self.pbar.update(1000)

        if self.scanned_count % self.step != 0:
            return

        if not a.tags:
            return

        try:
            self._extract_poi(
                a.tags,
                lambda: self.wkb_factory.create_multipolygon(a),
                "way" if a.from_way() else "relation",
                a.orig_id()
            )
        except Exception:
            pass

    def close(self):
        """Finalize and close handler"""
        self._flush()
        self.pbar.close()
        if self.writer:
            self.writer.close()
            self.writer = None

        # Cleanup
        self.columns.clear()


# ================= PROCESSING FUNCTIONS =================

def extract_pois(input_file, output_file, poi_batch_size, max_pois, step):
    """Extract POIs from OSM PBF file"""
    print(f"\n{'='*60}")
    print(f"EXTRACTING POIs: {input_file}")
    print(f"{'='*60}")

    file_size = os.path.getsize(input_file)
    estimated_elements = file_size // 5

    handler = POIHandler(
        output_file,
        total_elements=estimated_elements,
        batch_size=poi_batch_size,
        max_pois=max_pois,
        step=step
    )

    start_time = time.time()

    try:
        handler.apply_file(input_file, locations=True, idx="flex_mem")
    except LimitReached:
        print("\nMax POI limit reached.")
    finally:
        handler.close()
        elapsed = time.time() - start_time

        if os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"\n(!) Extracted {handler.total_pois:,} POIs in {elapsed:.1f}s")
            print(f"  Output: {output_file} ({size_mb:.2f} MB)")
            return handler.total_pois
        else:
            print(f"\n✗ No POIs extracted")
            return 0


def enrich_with_geocoding(input_file, output_file, geo_batch_size):
    """Add reverse geocoding information to POIs"""
    print(f"\n{'='*60}")
    print(f"ENRICHING: {input_file}")
    print(f"{'='*60}")

    # Initialize geocoder
    print("Loading geocoder database...")
    rg.search((0, 0))
    print("(!) Geocoder ready")

    parquet_file = pq.ParquetFile(input_file)
    writer = None
    processed_count = 0
    start_time = time.time()

    try:
        for i, batch in enumerate(parquet_file.iter_batches(batch_size=geo_batch_size)):
            # Load chunk
            df = batch.to_pandas()

            # Get coordinates
            coords = list(zip(df["lat"], df["lon"]))

            # Batch geocode
            geo_results = rg.search(coords, mode=2)
            geo_df = pd.DataFrame(geo_results)
            geo_df = geo_df.drop(columns=['lat', 'lon'])
            geo_df.columns = [f"addr_{col}" for col in geo_df.columns]

            # Merge
            enriched_df = pd.concat(
                [df.reset_index(drop=True), geo_df.reset_index(drop=True)],
                axis=1
            )

            # Write
            table = pa.Table.from_pandas(enriched_df)
            if writer is None:
                writer = pq.ParquetWriter(output_file, table.schema,
                                         compression='snappy')
            writer.write_table(table)

            processed_count += len(df)

            # Cleanup after each batch
            del df, geo_df, enriched_df, table, batch

            if (i + 1) % 10 == 0:
                print(f"  Processed batch {i+1}: {processed_count:,} rows")

    finally:
        # Cleanup
        if writer:
            writer.close()
            writer = None

        del parquet_file

        elapsed = time.time() - start_time

        if os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"\n(!) Enriched {processed_count:,} POIs in {elapsed:.1f}s")
            print(f"  Output: {output_file} ({size_mb:.2f} MB)")
        else:
            print(f"\n(X) Enrichment failed")
            processed_count = 0

    return processed_count


def process_file(input_file, output_dir, poi_batch_size, geo_batch_size,
                max_pois, step, keep_raw, resume):
    """Process a single OSM file through the full pipeline"""
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"(X) Error: File not found: {input_file}")
        return False

    # Generate output filenames
    base_name = input_path.stem
    raw_output = output_dir / f"{base_name}_pois_raw.parquet"
    enriched_output = output_dir / f"{base_name}_pois_enriched.parquet"

    # Resume check
    if resume and enriched_output.exists():
        size_mb = enriched_output.stat().st_size / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"(!) Skipping (already exists): {enriched_output.name} ({size_mb:.2f} MB)")
        print(f"{'='*60}")
        return True

    print(f"\n{'#'*60}")
    print(f"PROCESSING: {input_path.name}")
    print(f"{'#'*60}")

    # Extract POIs
    poi_count = extract_pois(str(input_path), str(raw_output), poi_batch_size,
                            max_pois, step)

    if poi_count == 0:
        print(f"(X) Skipping enrichment (no POIs found)")
        return False

    # Enrich with geocoding
    enriched_count = enrich_with_geocoding(str(raw_output), str(enriched_output),
                                          geo_batch_size)

    # Cleanup raw file
    if not keep_raw and raw_output.exists():
        raw_size_mb = raw_output.stat().st_size / (1024 * 1024)
        raw_output.unlink()
        print(f"\n(!) Cleaned up raw file ({raw_size_mb:.2f} MB)")

    success = enriched_count > 0
    print(f"\n{'='*60}")
    print(f"{'(!) COMPLETE' if success else '(X) FAILED'}: {input_path.name}")
    print(f"{'='*60}\n")

    return success


# ================= CLI =================

def main():
    parser = argparse.ArgumentParser(
        description="Extract POIs from OSM PBF files and enrich with geocoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "input_files",
        nargs="+",
        help="OSM PBF input file(s) to process"
    )

    parser.add_argument(
        "-o", "--output-dir",
        default=".",
        help="Output directory for parquet files"
    )

    parser.add_argument(
        "--poi-batch-size",
        type=int,
        default=500_000,
        help="Batch size for POI extraction (rows per write)"
    )

    parser.add_argument(
        "--geo-batch-size",
        type=int,
        default=100_000,
        help="Batch size for geocoding (rows per batch)"
    )

    parser.add_argument(
        "--max-pois",
        type=int,
        default=None,
        help="Maximum number of POIs to extract (None for unlimited)"
    )

    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Process every Nth element (1 for all, 10 for every 10th, etc.)"
    )

    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep raw (non-enriched) parquet files"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files that already have enriched output"
    )

    args = parser.parse_args()

    # Validate POI mapping
    if not POI_MAPPING:
        print("(X) Error: POI_MAPPING is empty")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process files
    total_files = len(args.input_files)
    successful = 0
    failed = []
    skipped = 0

    overall_start = time.time()

    for i, input_file in enumerate(args.input_files, 1):
        print(f"\n{'#'*60}")
        print(f"FILE {i}/{total_files}")
        print(f"{'#'*60}")

        # Check if already done (for resume)
        input_path = Path(input_file)
        enriched_output = output_dir / f"{input_path.stem}_pois_enriched.parquet"

        if args.resume and enriched_output.exists():
            skipped += 1
            size_mb = enriched_output.stat().st_size / (1024 * 1024)
            print(f"(!) Skipping (already exists): {enriched_output.name} ({size_mb:.2f} MB)")
            continue

        success = process_file(
            input_file,
            output_dir,
            args.poi_batch_size,
            args.geo_batch_size,
            args.max_pois,
            args.step,
            args.keep_raw,
            args.resume
        )

        if success:
            successful += 1
        else:
            failed.append(input_file)

    overall_elapsed = time.time() - overall_start

    # Final summary
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")
    print(f"Processed: {successful}/{total_files} files")
    if skipped > 0:
        print(f"Skipped: {skipped} files (already exist)")

    if failed:
        print(f"\nFailed files:")
        for f in failed:
            print(f"  ✗ {f}")
        sys.exit(1)
    else:
        print(f"\n✓ All files processed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()