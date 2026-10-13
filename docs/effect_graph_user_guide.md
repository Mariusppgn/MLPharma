# Explorer DeepPharma

Ouvrir `effect_graph.html` dans un navigateur. Le fichier est autonome et ne requiert
ni Python ni connexion internet pour être consulté. Copier le HTML suffit pour
transférer cette visualisation et ses données embarquées ; il ne faut pas le publier
sans vérifier les droits de partage des données.

## Première visite

1. Commencer par **Carte des effets**, sans choisir de molécule.
2. Chaque rectangle représente un effet. Les noms sont traduits pour l'affichage ;
   **Libellés d'origine** permet de retrouver les termes anglais du dataset.
3. Déplacer le fond avec la souris, utiliser +/− ou régler la taille du texte.
   La grille de lecture conserve un zoom lisible : une partie de la carte peut être
   hors écran. **Tout voir** réduit le zoom pour afficher l'ensemble ; **Recentrer
   en lecture** revient à la vue lisible.
4. Cliquer sur un effet ou un lien pour ses indicateurs. La liste sous le graphe
   permet aussi de sélectionner les effets au clavier, sans manipuler la carte.

## Filtrer

Les filtres de population concernent toutes les vues exploratoires. Les filtres
de relations limitent les liens affichés, pas les paires utilisées dans les
fréquences. Par défaut, seuls les 40 liens les mieux classés par stabilité × force
sont visibles. Augmenter leur nombre dans **Relations affichées** si nécessaire.
Un nœud sans lien visible ne signifie pas absence de dépendance.

Le dataset ne fournit pas les noms des médicaments : les molécules sont identifiées
par un identifiant local et leur SMILES. Une molécule absente ne produit pas de
prédiction inventée. Les filtres chimiques et pharmacologiques se trouvent dans
des sections repliables. Réinitialiser enlève les restrictions de population.

## Lire les résultats

- **Fréquence** : proportion de paires filtrées où l'effet est annoté, pas incidence
  chez des patients.
- **PRR moyen positif** : moyenne des PRR strictement positifs fournis par le dataset.
- **Stabilité** : proportion de rééchantillonnages où un lien est retrouvé. Les
  estimations sont globales et ne sont pas recalculées après chaque filtre.
- **Poids** : association statistique, positive ou négative. Les flèches BIC sont
  des orientations de factorisation statistique, pas des mécanismes biologiques.
- **Résultats & méthodes** : scores hors apprentissage du benchmark, indépendants
  des filtres exploratoires. Ce benchmark n'est pas le score Kaggle global.

Les mentions CYP sont des extractions textuelles heuristiques. Aucun usage clinique
ou conseil thérapeutique n'est validé par cette interface.

## Exports et mise à jour

CSV : effets et liens visibles avec leurs valeurs précises. PNG : rendu du graphe.
SVG : diagramme vectoriel avec les mêmes positions et libellés ; les liens courbes
sont simplifiés en segments. La taille du texte choisie s'applique aux images.

Pour appliquer une modification d'interface aux sorties existantes, sans entraîner :

```powershell
.\.venv\Scripts\python.exe scripts\generate_graph_dashboard.py --run outputs\graph\effect_graph_v1
```

Le générateur actualise les diagnostics descriptifs et centralités, mais n'entraîne
aucun modèle et ne modifie ni les prédictions ni les scores de validation.
