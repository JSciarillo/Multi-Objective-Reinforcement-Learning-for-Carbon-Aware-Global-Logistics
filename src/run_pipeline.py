"""
Runs the full data-engineering pipeline in order:
  1. extract_network.py   — pull ~2,000 node DC graph from OSM
  2. elevation_data.py    — add SRTM elevation + road grade
  3. add_speeds.py        — time-of-day speed profiles
  4. carbon_calculate.py  — MOVES5 carbon weights (12 per edge)

Run from the repo root OR from the src/ directory:
    python src/run_pipeline.py
    cd src && python run_pipeline.py
"""

import subprocess
import sys
import os

STEPS = [
    ("extract_network.py",  "Network extraction    (OSM → dc_subgraph.graphml)"),
    ("elevation_data.py",   "Elevation & grade     (→ dc_subgraph_elev.graphml)"),
    ("add_speeds.py",       "Speed profiles        (→ dc_subgraph_speeds.graphml)"),
    ("carbon_calculate.py", "Carbon weights        (→ dc_subgraph_carbon.graphml)"),
]

def main():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    for script, description in STEPS:
        print(f"\n{'─'*55}")
        print(f"  {description}")
        print(f"{'─'*55}")
        result = subprocess.run(
            [sys.executable, os.path.join(src_dir, script)],
            cwd=src_dir,
        )
        if result.returncode != 0:
            print(f"\nPipeline failed at {script}. Fix the error above and re-run.")
            sys.exit(result.returncode)

    print("\n✓ Pipeline complete. Graph ready at data/dc_subgraph_carbon.graphml")
    print("  Next: python src/train_ppo.py")

if __name__ == "__main__":
    main()
