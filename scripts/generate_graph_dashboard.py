import argparse
from pathlib import Path

from deep_pharma.graph.visualization import generate_dashboard

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate offline effect graph HTML")
    parser.add_argument("--run", type=Path, required=True)
    generate_dashboard(parser.parse_args().run)
