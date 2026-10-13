# Réseau de dépendances des effets secondaires

Module de recherche indépendant du modèle Kaggle : aucune lecture du test, aucune
soumission, aucune promotion automatique. Les observations sont des annotations de
paires moléculaires, pas des dossiers patients ni des cas de pharmacovigilance.

## Exécution CPU

Depuis la racine, PowerShell :

```powershell
.\.venv\Scripts\python.exe scripts\train_effect_graph.py --config configs\effect_graph.yaml
.\.venv\Scripts\python.exe scripts\generate_graph_dashboard.py --run outputs\graph\effect_graph_v1
.\.venv\Scripts\python.exe -m pytest tests\test_effect_graph.py -q
```

Un run terminé n'est pas écrasé : changer `run_dir` pour une nouvelle expérience.
Le journal, le statut et les scores intermédiaires sont persistés. Le pipeline ne
reprend pas encore automatiquement les étapes longues interrompues ; le cache de
features est réutilisable lorsque sa signature correspond.

## Pipeline

Train + manifeste des folds → audit des paires canoniques → features symétriques
RDKit/PK/texte → screening MI → structures conditionnelles → bootstrap groupé →
benchmark prédictif indépendant/graphe → clusters exploratoires → HTML autonome.

- Les features RDKit/PK existantes sont réutilisées. Les fingerprints et textes
  sont projetés par hashing fixe pour limiter le coût CPU ; aucun transformateur
  supervisé n'est ajusté globalement pour la validation.
- La référence marginale calcule MI, phi, lift, odds ratio avec correction 0,5,
  intervalle OR approximatif et correction BH des tests marginaux.
- Le squelette conditionnel est inspiré de PC-stable, avec screening et profondeur
  bornée : il ne constitue pas une implémentation complète de PC/CPDAG.
- Les voisinages logistiques L1 conditionnent chaque effet aux covariables et aux
  effets voisins. Des interactions avec trois covariables modulent les poids.
  Ces modèles descriptifs voient les effets observés ; ils ne sont jamais utilisés
  tels quels comme prédictions hors-fold.
- Le réseau bayésien discret utilise une recherche gloutonne ajout/suppression BIC,
  avec deux parents maximum. Ses flèches expriment une factorisation statistique,
  pas une causalité ni une orientation biologique validée.
- Le bootstrap rééchantillonne des groupes de paires entiers. La fréquence de
  sélection, son intervalle Monte-Carlo Wilson et les quantiles des poids incluant
  les absences sont exportés. Vingt répétitions constituent une exploration, pas
  une garantie de contrôle des fausses découvertes. Les molécules partagées restent
  une source de dépendance non corrigée.
- Les clusters chimiques sont appris sur des descripteurs RDKit standardisés.
  Leurs graphes locaux sont exploratoires, éventuellement chevauchants et sans
  bootstrap local. Ils ne sont pas utilisés pour l'évaluation prédictive.

## Validation sans fuite

Le benchmark compare deux ensembles de têtes LightGBM binaires de même famille et
hyperparamètres fixés. Chaque structure prédictive est réapprise dans le seul
apprentissage externe. Les entrées des voisins sont des probabilités cross-fittées
par groupes dans l'apprentissage, et des probabilités prédites en validation.
Aucun label vrai de validation, aucun effet propre de la cible ni graphe appris
sur tout le train ne conditionne les prédictions de validation.

Protocoles : cinq folds des paires canoniques ; molécules disjointes cold/cold avec
purge des couples traversants ; scaffold-disjoint avec repli par molécule pour les
structures acycliques ; leave-one-drug-out pour les trois molécules les plus
représentées. Les partenaires d'un médicament laissé de côté peuvent rester connus.
Les autres protocoles sont des holdouts de robustesse, pas cinq folds supplémentaires.

Scores : AUPRC/AUROC, précision/rappel/F1 micro, Brier, ECE, précision/rappel@K,
rappel des effets rares et métriques par effet. Le seuil 0,5 n'est pas optimisé sur
la validation. Le PRR est décrit dans le graphe, mais ce benchmark ne régresse pas
le PRR et ne remplace pas le modèle global Kaggle retenu.

## Livrables et limites

`nodes.csv`, `edges.csv`, `bootstrap_edges.csv`, `cluster_graphs.json`,
`conditional_model.json`, `data_profile.json`, `evaluation.json`, prédictions NPZ,
`per_effect_metrics.csv`, `precision_recall.json`, `graph_metadata.json` et
`effect_graph.html` / `effect_graph_report.html`.

Les HTML embarquent Cytoscape et les données ; ils sont utilisables hors connexion.
Les descriptions textuelles et extractions CYP sont des indications heuristiques,
pas des annotations pharmacologiques validées. ATC, MedDRA/SOC, exposition, temps,
cas cliniques et constantes d'affinité absents ne sont pas inventés. Les motifs
non additifs sont des contrastes exploratoires non ajustés, sans validation causale.
Les p-values globales ne sont pas recalculées après un filtre interactif.

Ce travail ne fournit ni recommandations thérapeutiques, ni probabilités d'incidence
clinique, ni validation externe. Une utilisation scientifique requiert une analyse
des sources, des biais de reporting et des facteurs de confusion.

Références : [PC-stable](https://www.jmlr.org/papers/v15/colombo14a.html),
[stability selection](https://arxiv.org/abs/0809.2932),
[Cytoscape.js](https://js.cytoscape.org/).

## Coût et extensions

Pour 50 effets : 1 225 associations marginales ; au plus 300 arêtes candidates
avec six voisins par effet ; au plus cinq ensembles de conditionnement par arête
avec profondeur un et quatre voisins admissibles. Les 20 bootstraps réapprennent
50 modèles logistiques chacun. La recherche BIC s'arrête à 60 opérations et met en
cache les scores locaux. L'évaluation ajoute des cross-fits internes coûteux.

Les représentations supervisées existantes (embeddings relationnels, TF-IDF/SVD
ajustés, GNN) ne sont pas importées depuis des fits globaux, pour éviter une fuite.
Ce prototype CPU utilise leurs familles chimiques/PK/textuelles, mais pas toutes
leurs représentations apprises. Une ablation fold-local pourra les ajouter.

Restent expérimentaux ou non implémentés : orientation bootstrap des flèches BIC,
sensibilité prédictive à une grille d'hyperparamètres, graphes locaux validés,
motifs surreprésentés contre un modèle nul, clustering pharmacologique validé,
généralisation à des couples entièrement nouveaux depuis le HTML, régression PRR.
Le seuil de stabilité est interactif ; le choix d'un seuil ne démontre pas en soi
qu'une relation est vraie.
