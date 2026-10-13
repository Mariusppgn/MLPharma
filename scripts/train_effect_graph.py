import argparse
from pathlib import Path

from deep_pharma.graph.pipeline import run_effect_graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research statistical effect graph, train only")
    parser.add_argument("--config", type=Path, default=Path("configs/effect_graph.yaml"))
    run_effect_graph(parser.parse_args().config)
