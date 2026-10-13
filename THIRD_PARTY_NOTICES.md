# Third-party notices

The project's MIT license covers original DeepPharma code and documentation,
not third-party software or competition datasets.

## Cytoscape.js

The local effect-graph interface bundles Cytoscape.js 3.33.1 under its own MIT
license. Its original copyright and permission notice are preserved in
[`src/deep_pharma/graph/assets/CYTOSCAPE_LICENSE`](src/deep_pharma/graph/assets/CYTOSCAPE_LICENSE).
Source URLs and SHA-256 checksums are recorded in
[`vendor.json`](src/deep_pharma/graph/assets/vendor.json).

## Installed dependencies

RDKit, PyTorch, LightGBM, scikit-learn and the other dependencies listed in
`pyproject.toml` / `uv.lock` are installed separately and retain their respective
licenses. They are not relicensed by this repository.

## Kaggle datasets

Competition data, data-containing HTML exports, fitted models and predictions
are excluded from the public source release. Obtain the official files through
Kaggle and comply with the competition terms; the MIT license grants no rights
to those datasets.
