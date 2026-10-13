"""Generate the reproducible data-contract notebook with nbformat."""

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
            """# Deep-Pharma — contrat de données et première inspection

## tl;dr

Les fichiers officiels contiennent 15 284 lignes de train, 3 974 lignes de test, 25 variables
d'entrée et 101 cibles. Les cibles sont complètes, mais la sévérité est déséquilibrée et seules
trois classes sont observées. Des paires moléculaires répétées imposent une validation groupée
pour éviter une estimation artificiellement optimiste."""
        ),
        nbf.v4.new_markdown_cell(
            """## Contexte et méthodes

La compétition combine une classification multiclasse, une classification multilabel et une
régression masquée. Cette étape contrôle le schéma, les volumes, les valeurs manquantes, les
doublons et quelques distributions de cibles. Elle n'introduit ni données externes, ni
feature engineering, ni choix de modèle.

### Hypothèses clés

- `sample_submission.csv` est l'autorité pour l'ordre exact des colonnes et des identifiants.
- Les préfixes `Target_Binary_` et `Target_PRR_` identifient les deux familles de 50 cibles.
- Toute contradiction entre la description Kaggle et les CSV est conservée comme avertissement."""
        ),
        nbf.v4.new_markdown_cell("## Données"),
        nbf.v4.new_code_cell(
            """from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

from deep_pharma.constants import BINARY_PREFIX, PRR_PREFIX, SEVERITY_COLUMN
from deep_pharma.data_contract import inspect_data_contract
from deep_pharma.validation import frame_pair_keys

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.figsize": (10, 5), "axes.titleweight": "bold"})

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "pyproject.toml").is_file():
    PROJECT_ROOT = PROJECT_ROOT.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
required_files = [RAW_DIR / name for name in ("train.csv", "test.csv", "sample_submission.csv")]
HAS_DATA = all(path.is_file() for path in required_files)

print(f"Racine : {PROJECT_ROOT}")
print(f"Données officielles présentes : {HAS_DATA}")
if not HAS_DATA:
    display(
        Markdown(
            "**Action requise :** accepter les règles Kaggle, s'authentifier, "
            "puis lancer `scripts/download_data.ps1`."
        )
    )"""
        ),
        nbf.v4.new_markdown_cell("### 1. Valider le contrat"),
        nbf.v4.new_code_cell(
            """contract = inspect_data_contract(RAW_DIR)
print(contract.to_json())"""
        ),
        nbf.v4.new_markdown_cell("### 2. Charger des aperçus bornés"),
        nbf.v4.new_code_cell(
            """if HAS_DATA:
    train = pd.read_csv(RAW_DIR / "train.csv", low_memory=False)
    test = pd.read_csv(RAW_DIR / "test.csv", low_memory=False)
    sample = pd.read_csv(RAW_DIR / "sample_submission.csv", low_memory=False)
    display(train.head(3))
    display(
        pd.DataFrame(
            {
                "fichier": ["train.csv", "test.csv", "sample_submission.csv"],
                "lignes": [len(train), len(test), len(sample)],
                "colonnes": [train.shape[1], test.shape[1], sample.shape[1]],
            }
        )
    )
else:
    display(Markdown("Aucun aperçu : les CSV officiels ne sont pas encore présents."))"""
        ),
        nbf.v4.new_markdown_cell("### 3. Contrôler la structure des paires"),
        nbf.v4.new_code_cell(
            """if HAS_DATA:
    train_pair_key = frame_pair_keys(train)
    test_pair_key = frame_pair_keys(test)
    pair_sizes = train_pair_key.value_counts()
    display(
        pd.DataFrame(
            {
                "contrôle": [
                    "Groupes de paires répétés dans train",
                    "Lignes train appartenant à une paire répétée",
                    "Paires canoniques communes à train et test",
                ],
                "valeur": [
                    int((pair_sizes > 1).sum()),
                    int(train_pair_key.duplicated(keep=False).sum()),
                    len(set(train_pair_key) & set(test_pair_key)),
                ],
            }
        )
    )
else:
    display(Markdown("Contrôle différé jusqu'au téléchargement des données."))"""
        ),
        nbf.v4.new_markdown_cell("## Résultats"),
        nbf.v4.new_markdown_cell("### 4. Distribution de la sévérité"),
        nbf.v4.new_code_cell(
            """if HAS_DATA:
    severity_counts = (
        train[SEVERITY_COLUMN]
        .value_counts(dropna=False)
        .rename_axis("Severity")
        .reset_index(name="n")
    )
    severity_counts["part"] = severity_counts["n"] / len(train)
    display(severity_counts)
    axis = sns.barplot(data=severity_counts, x="Severity", y="part", color="#3B82F6")
    axis.set(
        title="Répartition des classes de sévérité — train officiel",
        xlabel="Classe",
        ylabel="Part des lignes",
    )
    axis.bar_label(
        axis.containers[0],
        labels=[f"{value:.1%}" for value in severity_counts["part"]],
    )
    plt.tight_layout()
    plt.show()
else:
    display(Markdown("Graphique différé jusqu'au téléchargement des données."))"""
        ),
        nbf.v4.new_markdown_cell("### 5. Prévalence des cibles binaires"),
        nbf.v4.new_code_cell(
            """if HAS_DATA:
    binary_columns = [column for column in train if column.startswith(BINARY_PREFIX)]
    prevalence = train[binary_columns].mean().sort_values(ascending=False)
    display(prevalence.describe().to_frame("prévalence"))
    top_prevalence = prevalence.head(15).sort_values()
    axis = top_prevalence.plot.barh(color="#10B981")
    axis.set(
        title="15 effets indésirables les plus fréquents — train officiel",
        xlabel="Prévalence",
        ylabel="Cible",
    )
    plt.tight_layout()
    plt.show()
else:
    display(Markdown("Graphique différé jusqu'au téléchargement des données."))"""
        ),
        nbf.v4.new_markdown_cell("### 6. Valeurs PRR strictement positives"),
        nbf.v4.new_code_cell(
            """if HAS_DATA:
    prr_columns = [column for column in train if column.startswith(PRR_PREFIX)]
    positive_prr = train[prr_columns].stack()
    positive_prr = positive_prr[positive_prr > 0]
    display(positive_prr.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_frame("PRR positif"))
    axis = sns.histplot(positive_prr, bins=50, color="#F59E0B")
    axis.set(
        title="Distribution des PRR positifs — train officiel",
        xlabel="PRR",
        ylabel="Nombre d'observations",
    )
    plt.tight_layout()
    plt.show()
else:
    display(Markdown("Graphique différé jusqu'au téléchargement des données."))"""
        ),
        nbf.v4.new_markdown_cell(
            """## Conclusions et prochaine étape

Le premier choix de pipeline doit être le protocole de validation, avant les représentations et
les modèles. Un découpage aléatoire par ligne laisserait des paires répétées traverser les folds.
Nous comparerons donc au minimum une validation groupée par paire moléculaire non ordonnée et une
validation plus sévère groupée par molécule, tout en stratifiant autant que possible la sévérité
et la présence d'au moins un effet indésirable."""
        ),
    ]

    output_path = Path("notebooks/00_data_contract.ipynb")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
