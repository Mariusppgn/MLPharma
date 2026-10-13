"""Generate the reproducible feature-analysis notebook."""

from pathlib import Path

import nbformat as nbf


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python (deep-pharma-starter)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Deep-Pharma — analyse des features avant modélisation

## tl;dr

- Le test est presque entièrement **warm/warm** : 3 935 paires sur 3 974.
- Les 50 cibles binaires valent exactement `1(PRR > 0)` dans les 764 200 cellules observées.
- Les cibles restent apprises séparément dans la première baseline afin de comparer leur
  cohérence hors échantillon.
- Les features retenues combinent Morgan, descripteurs RDKit, TF-IDF/SVD et variables PK/CYP.
- `Warning` et `Toxicity` seront soumis à une ablation dédiée à cause du risque de raccourci
  sémantique."""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

Cette analyse sert à décider la représentation du premier modèle. Les graphiques utilisent
uniquement les cibles du train et l'audit produit avant entraînement. Les cibles ne deviennent
jamais des variables d'entrée.

### Key Assumptions

- `Pair_ID` est un identifiant, jamais une feature.
- La sévérité est traitée comme nominale : aucun ordre clinique n'est inventé.
- Les statistiques apprises par le modèle restent ajustées dans le train de chaque fold.
- Les distributions de cibles ci-dessous sont diagnostiques et ne sont pas calculées sur le test."""
        ),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

from deep_pharma.constants import BINARY_PREFIX, PRR_PREFIX, SEVERITY_COLUMN

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"blue": "#2563EB", "gold": "#D97706", "pink": "#DB2777", "grey": "#64748B"}
plt.rcParams.update({"figure.figsize": (10, 5), "axes.titleweight": "bold"})

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "pyproject.toml").is_file():
    PROJECT_ROOT = PROJECT_ROOT.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
AUDIT_PATH = PROJECT_ROOT / "reports" / "data_audit.json"

audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
header = pd.read_csv(RAW_DIR / "train.csv", nrows=0)
binary_columns = [column for column in header if column.startswith(BINARY_PREFIX)]
prr_columns = [column for column in header if column.startswith(PRR_PREFIX)]
train_targets = pd.read_csv(
    RAW_DIR / "train.csv",
    usecols=[SEVERITY_COLUMN, *binary_columns, *prr_columns],
)
print(f"Train analysé : {len(train_targets):,} lignes, {len(binary_columns)} événements")"""
        ),
        nbf.v4.new_markdown_cell("## Results"),
        nbf.v4.new_markdown_cell("### 1. Composition warm/cold du test"),
        nbf.v4.new_code_cell(
            """regime_counts = pd.Series(audit["chemistry"]["test_regime_counts"])
regime_counts = regime_counts.reindex(["warm/warm", "warm/cold", "cold/cold"], fill_value=0)
axis = regime_counts.plot.barh(color=PALETTE["blue"])
axis.set(
    title="Régimes moléculaires des 3 974 paires test",
    xlabel="Nombre de paires",
    ylabel="Régime par rapport au train",
)
axis.bar_label(axis.containers[0], fmt="%d")
plt.tight_layout()
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """Le test contient 99,0 % de paires warm/warm. La validation par paire est donc la
plus proche du cas d'usage, tandis que le holdout cold-drug reste un stress-test séparé."""
        ),
        nbf.v4.new_markdown_cell("### 2. Déséquilibre de Severity"),
        nbf.v4.new_code_cell(
            """severity_rates = train_targets[SEVERITY_COLUMN].value_counts(normalize=True)
axis = severity_rates.plot.bar(color=PALETTE["gold"])
axis.set(
    title="Distribution de Severity — train officiel",
    xlabel="Classe",
    ylabel="Part des 15 284 lignes",
    ylim=(0, 0.75),
)
axis.bar_label(axis.containers[0], labels=[f"{value:.1%}" for value in severity_rates])
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """La classe Minor ne représente que 4,1 % du train. Le classifieur Severity utilise
donc une pondération équilibrée et la métrique macro-F1."""
        ),
        nbf.v4.new_markdown_cell("### 3. Prévalence des 50 effets indésirables"),
        nbf.v4.new_code_cell(
            """binary_prevalence = train_targets[binary_columns].mean()
axis = binary_prevalence.plot.hist(bins=12, color=PALETTE["pink"], edgecolor="white")
axis.set(
    title="Distribution des prévalences — 50 cibles binaires",
    xlabel="Part de lignes positives par cible",
    ylabel="Nombre de cibles",
)
plt.tight_layout()
plt.show()
display(
    pd.concat(
        [binary_prevalence.nsmallest(5), binary_prevalence.nlargest(5)]
    ).rename("prévalence").to_frame().style.format({"prévalence": "{:.1%}"})
)"""
        ),
        nbf.v4.new_markdown_cell(
            """Les prévalences sont comprises approximativement entre 18 % et 27 %. Il n'y a pas
de cible quasi vide, mais un seuil unique de 0,5 n'est pas nécessairement optimal. Les seuils
sont appris par cible sur les prédictions out-of-bag du train du fold."""
        ),
        nbf.v4.new_markdown_cell("### 4. Distribution positive des PRR"),
        nbf.v4.new_code_cell(
            """prr_values = train_targets[prr_columns].to_numpy(dtype=float)
positive_prr = prr_values[prr_values > 0]
axis = sns.histplot(np.log1p(positive_prr), bins=50, color=PALETTE["blue"])
axis.set(
    title="Distribution des PRR positifs — échelle log1p",
    xlabel="log(1 + PRR)",
    ylabel="Nombre d'observations positives",
)
plt.tight_layout()
plt.show()
display(
    pd.Series(positive_prr)
    .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    .to_frame("PRR positif")
)"""
        ),
        nbf.v4.new_markdown_cell(
            """La forte asymétrie justifie l'apprentissage de `log1p(PRR)` avec retour par
`expm1`, sans clipping arbitraire au maximum du train."""
        ),
        nbf.v4.new_markdown_cell("### 5. Severity et charge d'effets indésirables"),
        nbf.v4.new_code_cell(
            """diagnostic = pd.DataFrame({
    "Severity": train_targets[SEVERITY_COLUMN],
    "Nombre d'effets positifs": train_targets[binary_columns].sum(axis=1),
})
order = [label for label in ("Minor", "Moderate", "Major") if label in set(diagnostic["Severity"])]
axis = sns.boxplot(
    data=diagnostic,
    x="Severity",
    y="Nombre d'effets positifs",
    order=order,
    color="#93C5FD",
    showfliers=False,
)
axis.set(title="Charge d'effets positifs selon Severity — diagnostic sur le train")
plt.tight_layout()
plt.show()
display(diagnostic.groupby("Severity")["Nombre d'effets positifs"].describe().round(2))"""
        ),
        nbf.v4.new_markdown_cell(
            """Cette association est uniquement descriptive. Les vraies cibles binaires ne sont
jamais utilisées pour prédire Severity ; une éventuelle chaîne future devra utiliser des
prédictions OOF."""
        ),
        nbf.v4.new_markdown_cell("## Takeaways"),
        nbf.v4.new_markdown_cell(
            """1. La représentation de paire sera symétrique et les doublons resteront dans le
   même fold.
2. Le modèle combine structure, texte et variables PK/CYP, avec tout preprocessing appris
   fold par fold.
3. Binary et PRR sont cohérents au seuil zéro, mais restent modélisés séparément dans cette
   baseline pour mesurer la qualité réelle des deux tâches.
4. Le score warm-pair est le critère de sélection principal ; le cold-drug mesure une
   généralisation plus sévère et n'est pas mélangé au score principal.
5. Une ablation sans Warning/Toxicity est nécessaire avant d'attribuer le gain aux textes."""
        ),
    ]
    output_path = Path("notebooks/01_feature_analysis.ipynb")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
