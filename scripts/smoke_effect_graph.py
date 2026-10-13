"""Small train-only pilot; never overwrite official data or fold manifests."""

from pathlib import Path

import pandas as pd
import yaml

from deep_pharma.graph.pipeline import run_effect_graph

if __name__ == "__main__":
    folder = Path("work/effect_graph_smoke")
    folder.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path("configs/effect_graph.yaml").read_text())
    manifest = pd.read_csv(config["manifest"])
    chosen = manifest.pair_group.drop_duplicates().sample(n=600, random_state=2026)
    manifest = manifest[manifest.pair_group.isin(chosen)].sort_values("row_index").copy()
    frame = pd.read_csv(config["train"], low_memory=False).iloc[manifest.row_index].copy()
    frame.to_csv(folder / "train.csv", index=False)
    manifest["row_index"] = range(len(manifest))
    manifest.to_csv(folder / "folds.csv", index=False)
    config.update(
        train=str(folder / "train.csv"),
        manifest=str(folder / "folds.csv"),
        run_dir="outputs/graph/effect_graph_smoke_v1",
    )
    config["bootstrap"]["iterations"] = 2
    config["structure"].update(screening_neighbors=3, bic_max_steps=5)
    config["evaluation"].update(
        protocols=["pair_grouped"], outer_folds=[0], baseline_trees=5, min_rows=20
    )
    path = folder / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_effect_graph(path)
