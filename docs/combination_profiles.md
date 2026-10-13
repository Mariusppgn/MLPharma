# Comparateur descriptif de combinaisons DeepPharma

## Ouvrir directement dans le navigateur (recommandé)

Double-cliquer sur **`outputs/combination_profiles/DeepPharma_Combinaisons.html`**. Le fichier
autonome contient les données et les scripts : aucun serveur, installation ou accès Internet.
Utiliser un navigateur récent disposant de la décompression gzip intégrée (Chrome, Edge, Firefox).
Le fichier fait environ 15 Mo avec les structures 2D ; patienter quelques secondes à l'ouverture
pour la décompression.

Après le choix de **A**, les suggestions de **B** ne contiennent que ses partenaires canoniques
présents dans le train. Si A change, un ancien choix B incompatible est remplacé par un partenaire
valide. La comparaison rejette aussi une saisie manuelle de B incompatible. Les deux ordres A/B
sont reconnus ; les variantes `Pair_ID` du même couple restent disponibles.

Le HTML embarque un instantané du train et des OOF vérifiés, jamais le test ni des profils externes.
Les chiffres contextuels restent descriptifs, sans interprétation comme monothérapie. Le fichier
peut être déplacé seul ; il ne se met pas à jour automatiquement. Pour le régénérer :

```powershell
.\.venv\Scripts\python.exe scripts/generate_combination_profiles_html.py
```

Ou tâche VS Code **Deep-Pharma: générer le HTML autonome des combinaisons**. Le calcul local de
l'affichage hors ligne est vérifié contre Python ; les cellules manquantes restent nulles et les
références probabilistes restent indisponibles faute de profils monothérapie documentés.

### Structures topologiques des molécules

Après la sélection d'un couple valide, les structures 2D de A et B apparaissent côte à côte.
Elles se mettent à jour au changement de sélection et à l'inversion A/B. **Agrandir A/B** ouvre
une vue avec zoom de 100 à 300 % ; la zone peut défiler pour examiner les grandes molécules.
**SVG de A/B** télécharge séparément le dessin vectoriel correspondant.

RDKit dessine directement les SMILES canoniques du train : cycles, liaisons multiples, charges,
fragments/sels et stéréochimie renseignée. Les carbones et les hydrogènes usuels sont implicites.
Le nombre d'atomes affiché compte les atomes explicites du graphe SMILES, pas tous les hydrogènes
implicites de la formule brute. Les propriétés et le dessin partagent le même identifiant moléculaire.
Les 958 dessins sont embarqués dans le HTML, sans service externe ni réentraînement.

Ce sont des dessins topologiques 2D, **pas des conformations 3D**, ni un complexe de liaison A+B.
La stéréochimie non renseignée n'est pas inventée. Le dessin ne confirme pas l'exactitude des textes
pharmacologiques du fichier. Le générateur est `combination_profiles/structures.py`, avec version
RDKit conservée dans les métadonnées.

## Variante serveur dans VS Code

Dans le terminal PowerShell à la racine du projet :

```powershell
.\scripts\run_combination_profiles.ps1
```

Puis ouvrir **http://127.0.0.1:8766**. Arrêter avec Ctrl+C. Aucun crédit ChatGPT, téléchargement,
entraînement ou accès Kaggle n'est nécessaire. Le serveur reste sur la boucle locale, sans exposition
réseau. Alternative :

```powershell
.\.venv\Scripts\python.exe scripts/serve_combination_profiles.py --config configs/combination_profiles.yaml
.\.venv\Scripts\python.exe scripts/serve_combination_profiles.py --audit-only
.\.venv\Scripts\python.exe -m pytest tests/test_combination_profiles.py
```

Cette variante utilise un petit serveur Python local pour la préparation et les calculs ; son fichier
source `assets/index.html` ne s'ouvre pas seul. Pour un double-clic, utiliser le HTML autonome ci-dessus.
Aucun service cloud, CDN ou dépendance supplémentaire.

## Ce qui existe réellement

Source : `data/raw/train.csv`, 15 284 lignes, 958 molécules canoniques, 50 effets. L'audit de démarrage
est enregistré dans `outputs/combination_profiles/source_audit.json` avec les SHA256 des sources.
14 784 couples canoniques ; 489 ont plusieurs lignes. Les variantes sont sélectionnées par `Pair_ID`,
jamais moyennées pour le profil observé du couple.

Les données ne fournissent **aucun profil monothérapie**, nom canonique fiable, classe ATC,
classification SOC/PT MedDRA, effectif patient, période documentée ni intervalle de confiance clinique.
Les libellés français sont des traductions d'affichage, pas une annotation MedDRA.
Les textes pharmacologiques peuvent varier et même être incohérents avec la structure : les vérifier
avant toute interprétation médicale. L'application affiche la ligne source et le nombre de variantes.

### Trois lectures complémentaires

1. **Annotations observées 0/1** : cibles binaires telles que fournies. Zéro signifie non-détection
   dans ce fichier, pas absence biologique. A+B correspond à une seule ligne source sélectionnée.
2. **PRR bruts observés** : scores de disproportionalité, pas des incidences ou probabilités.
   Aucun PRR prédit n'est fourni par le benchmark graph binaire réutilisé ici.
3. **Probabilités OOF de recherche** : les cinq fichiers `evaluation_pair_grouped_*.npz`
   du run `outputs/graph/effect_graph_v1`, colonne `graph`. Ce modèle binaire de recherche n'est
   **pas l'ensemble retenu pour Kaggle**. Affichage seulement si SHA train, couverture, Pair_ID,
   formes, bornes et folds/couples canoniques sont vérifiés. Pas de nouvel entraînement et pas
   d'inférence inventée pour les couples absents. Il s'agit d'une probabilité d'annotation du dataset,
   pas d'une incidence clinique ; une calibration clinique n'est pas établie.

## Sans A/B seuls : contexte de coadministration et textes

La case « autres couples » est active par défaut pour la lecture binaire. Pour chaque molécule :

\[
 f_i(A;\neg AB)=\frac{\#\{\text{autres couples canoniques contenant A, annotés positifs pour }i\}}
 {\#\{\text{autres couples canoniques contenant A avec annotation non conflictuelle pour }i\}}.
\]

Toutes les lignes du couple étudié sont exclues, y compris dans l'ordre inverse. Un couple canonique
compte une fois. Des annotations contradictoires entre ses doublons rendent ce couple manquant
pour l'effet concerné. Le support `n` est le **nombre de couples exploitables**, jamais de patients ou
de déclarations. Dénominateur nul = non calculable. Cette fréquence décrit le corpus, pas A seul.
A/B sont affichés en %, A+B en annotation 0/1 : aucune soustraction entre ces objets ni référence
d'indépendance. Les catégories contextuelles portent des noms distincts, sans « émergence ».

Les fiches juxtaposent mécanisme, pharmacodynamie, métabolisme, CYP450, avertissements et les autres
champs disponibles. Les propriétés chimiques réutilisent RDKit du projet. Les mentions CYP communes
sont des correspondances textuelles exactes : aucune déduction automatique de rôle inhibiteur,
inducteur ou substrat, de nom de médicament, ni d'interaction clinique.

**Limite fondamentale** : des biais de choix des partenaires, d'indication, de déclaration et de
population empêchent d'interpréter les contextes comme les effets de A et B seuls. Ce module n'est
ni un modèle causal, ni un estimateur validé de synergie. Aucun résultat n'entre dans la sélection
de modèles ou les soumissions Kaggle. Aucun fichier test n'est ouvert.

## Vrais profils documentés ultérieurs (optionnel)

Pour une étude distincte, renseigner `profiles:` dans la configuration avec un CSV long contenant :

| Colonne | Définition |
|---|---|
| molecule_a, molecule_b | Identifiants `mol_...` du catalogue ; B vide pour un profil individuel |
| effect | Identifiant original de l'effet, ex. `Headache` |
| kind | `probability`, `prr` ou `annotation` |
| value | Valeur dans son échelle ; vide = manquante, pas zéro |
| source | Référence de source obligatoire |
| comparability_id | Identifiant du protocole de comparabilité documenté |
| meaning, population, period | Signification, population et période, explicites et identiques |
| n, ci_low, ci_high | Support et IC fournis par la source, optionnels ; aucun IC fabriqué |
| soc, pt | Métadonnées documentées optionnelles ; pas de mapping automatique |

Un seul enregistrement par couple/effet/type, doublons rejetés. L'identité du contexte est une
**déclaration méthodologique de l'auteur du fichier**, pas une validation clinique par l'application.
Les profils du couple importés restent dans leur source distincte, sans mélange avec le train.

Seulement si trois probabilités comparables sont disponibles et la case d'indépendance explicitement
acceptée :

\[
p_i^{ref}=1-(1-p_i(A))(1-p_i(B)),\qquad
\Delta_i^{pp}=100\bigl(p_i(A+B)-p_i^{ref}\bigr).
\]

Jamais de référence d'union pour les PRR, annotations ou fréquences de couples. Pour des PRR
documentés comparables, les écarts bruts AB−A/AB−B et log-ratios naturels ne sont affichés que sur
valeurs disponibles ; les log-ratios exigent des PRR strictement positifs. Pas de p-value inférée.

## Seuils, filtres et catégories

Probabilité/annotation/fréquence : valeur ≥ seuil (défaut 0,5). PRR : valeur > seuil (défaut 0).
Ce sont des **conventions descriptives configurables**, pas des seuils universels de significativité.
Un support minimal >0 rend les détections non calculables si le support est inconnu/insuffisant.
La marge vaut 0,05 par défaut : 5 points de probabilité, ou 0,05 unité PRR dans la vue PRR.

Partagé/A-spécifique/B-spécifique exigent deux profils individuels documentés comparables.
Émergent exige trois profils comparables, les deux individuels sous le seuil et AB au-dessus.
Renforcé/atténué exige un delta de probabilité calculable hors marge ; pour les PRR comparables,
AB doit dépasser max(A,B)+marge / être sous min(A,B)−marge. Les catégories se recoupent et ne
décrivent pas une causalité. Les contextes ont leurs propres catégories. Une donnée manquante
n'est jamais traitée comme une absence.

Jaccard est calculé sur les effets filtrés dont les deux détections sont connues et comparables :
\(J=|E_A\cap E_B|/|E_A\cup E_B|\). Union vide, données incomplètes ou incompatibles = non calculable.
En mode contextuel il compare les ensembles **des contextes**, pas les monothérapies.

Tous les filtres sont combinables. Les limites min/max portent sur **A+B seulement**, et les écarts
sur les deltas probabilistes calculables. Les menus SOC/PT, support patient et IC restent désactivés
quand la source ne les fournit pas. Les vues barres/heatmap/table/synthèse/exports partagent exactement
les mêmes lignes. Les vues delta/scatter sont indisponibles si aucun delta n'est calculable.
Le CSV conserve les valeurs brutes et les types, le SHA train, la ligne source, le seuil et les filtres ;
le JSON conserve l'ensemble des résultats et de la provenance. Les SVG/PNG sont des graphiques de
recherche, pas une preuve de validation clinique.

## Architecture et garde-fous

```text
train + OOF vérifiés + profils locaux optionnels
                    ↓
datastore.py : canonicalisation existante, provenance, exclusion/dédoublonnage
                    ↓
statistics.py + classification.py : valeurs nullables, comparabilité, filtres, synthèse
                    ↓
api.py : HTTP 127.0.0.1, lecture seule, cache borné de 128 requêtes
                    ↓
HTML/CSS/JS + SVG : tableau = graphiques = exports
```

Caches par instance bornés à 256 fiches/contextes ; aucune installation lourde. Rejet des origines
distantes et des hôtes inattendus, pas d'endpoint de fichiers arbitraires/upload ni CDN.
Le module graph et les pipelines d'entraînement restent indépendants et inchangés.

## Vérifications réalisées le 8 octobre 2026

- Suite Python complète : **110 tests passés**, dont 20 pour ce module ; Ruff du projet passé.
- Smoke test sur le vrai train : 958 molécules, 50 effets, probabilités OOF disponibles et vérifiées.
- Contrôles JavaScript en VM : correspondance tableau/graphiques/CSV, sélection, données manquantes,
  états vides, protection CSV, désactivation des références non calculables. Barres et heatmap sur
  les données réelles ; delta/scatter sur des fixtures de test explicitement synthétiques uniquement.
- Les quatre SVG générés passent la validation XML. API et HTML répondent HTTP 200, y compris
  sur le service réellement lancé ; les requêtes d'hôte/origine distants sont refusées.
- **Limite de vérification** : aucun navigateur automatisable n'était disponible. Le rendu réel
  desktop/mobile, la navigation au clavier dans le navigateur et le téléchargement PNG nécessitent
  encore une vérification visuelle par l'utilisateur. Les tests en VM ne remplacent pas cette revue.

Pour les lancements avec le script/VS Code, utiliser Ctrl+C dans le terminal pour arrêter le serveur.
Redémarrer le serveur après toute modification des fichiers sources pour renouveler les caches.

### Mise à jour : choix des partenaires et HTML autonome

Suite complète : **113 tests Python passés** et Ruff du projet passé. Le HTML autonome réel et ses
scripts embarqués ont été exécutés en VM : chargement gzip sans `fetch`, initialisation de l'interface,
119 cas de parité avec Python (17 couples, variantes, inversion, PRR, OOF, seuils/support et filtres),
et correspondance des listes de partenaires sur les 14 784 couples canoniques. 21 groupes de
contrôles frontend passent également. Ces tests ne remplacent pas une revue visuelle : aucun
navigateur automatisable n'était disponible lors de cette livraison.
