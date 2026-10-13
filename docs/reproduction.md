# Reproduire le pipeline

Le dépôt distribue le **code et les configurations**, pas un service de
prédiction prêt à l'emploi ni des poids entraînés. Les résultats historiques
ne constituent pas une promesse de reproduction bit à bit sur toute machine.
Les CSV et les artefacts ci-dessous doivent être recréés localement.

## Installation

Installer Python 3.12 et uv, puis, depuis la racine :

```powershell
uv sync --python 3.12 --locked
.\scripts\check.ps1
.\scripts\auth_kaggle.ps1
.\scripts\download_data.ps1
.\scripts\run_validation.ps1
```

L'acceptation des règles Kaggle est personnelle. Ne déposer aucun identifiant
d'accès dans Git. Les scripts utilisent `work/` pour les caches locaux.
La première commande d'authentification nécessite un accès Internet.

## Branches historiques et dépendances

Exécuter les commandes dans cet ordre. Les benchmarks écrivent leurs métriques
et leur progression dans `artifacts/runs/` et `outputs/`. Ce sont des calculs
CPU potentiellement longs ; les quick/smoke configs ne reproduisent pas les
scores complets.

```powershell
# Baseline et cache de features.
.\scripts\run_model.ps1
# OOF PRR de référence : deux Huber, hurdle et post-traitements comparés.
.\scripts\run_prr_oof.ps1
# Binary relationnel : branche C du benchmark.
.\scripts\run_four_branch_benchmark.ps1
# Candidat parent : lockbox distinct, puis CSV seulement si amélioration.
.\scripts\run_lockbox_candidate.ps1 -FinalizeIfImproved

# Artefacts historiques requis par les évaluations Severity/activité.
.\scripts\run_rupture_benchmark.ps1
.\scripts\run_prr_rupture_v2.ps1
.\.venv\Scripts\deep-pharma.exe run-severity-priors-benchmark
.\.venv\Scripts\deep-pharma.exe run-row-activity-benchmark

# Remplacer Severity, puis appliquer la porte d'activité Binary.
.\.venv\Scripts\deep-pharma.exe finalize-severity-priors-candidate
.\.venv\Scripts\deep-pharma.exe finalize-row-activity-candidate
.\.venv\Scripts\deep-pharma.exe validate-submission outputs\submission_severity_activity_v1.csv
```

Les scripts de finalisation possèdent leurs propres garde-fous : un échec ou
une absence de gain ne doit pas être contourné. Les benchmarks GNN/PRR v2 sont
des prérequis de certaines comparaisons historiques, **pas les sorties PRR ou
Binary retenues**. Lire les `paths` YAML si l'on change les noms de runs.
Le fichier `severity_activity_v1/metrics.json` et le registre de l'exécution
historique ne sont pas fournis : `scripts/verify_final_pipeline.py` ne fonctionne
que dans un workspace qui possède ces artefacts de provenance.

## Validation plus stricte pour les prochaines recherches

L'OOF PRR historique sélectionne les checkpoints sur le fold externe. Pour les
nouveaux essais, utiliser le workflow imbriqué, qui sélectionne les paramètres
dans l'outer-fit avant de scorer l'outer-validation :

```powershell
.\.venv\Scripts\deep-pharma.exe run-nested-cpu-optimization --help
.\.venv\Scripts\deep-pharma.exe run-nested-cpu-optimization --config configs\nested_cpu_optimization.yaml
```

Ce workflow est expérimental et ne remplace pas automatiquement le candidat
retenu. Ne pas comparer un score smoke/partiel au score complet ; ne jamais
choisir des hyperparamètres à partir du test officiel.

## Interfaces de recherche

Voir [combination_profiles.md](combination_profiles.md) et
[effect_graph_user_guide.md](effect_graph_user_guide.md).
Un HTML autonome généré embarque les données locales : ne pas le mettre en ligne.
Les notebooks peuvent être recréés à partir des scripts `generate_*notebook.py`.
