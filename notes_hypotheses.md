# Hypothèses et choix de réplication (Phase 1)

Réplication du Tableau 2 de Sugishita, Grothey & McKinnon (2024), engagement des unités
à 200 générateurs (`data/generators/out/200_0_1_w.csv`, capacité totale 34 344 MW).
Le protocole des instances suit `sucpy/problems/uc.py` (`load_demand_data`, `_sample_demand`).
Code : `uc/instances.py`, `uc/uc_bound.py`, `experiments/phase1.py`.

## Données de demande

- **Colonne ND** (National Demand) du fichier NESO `data/demand/out/demand.csv`.
- **Années 2017 à 2019 seulement.** Les auteurs avaient aussi le début de 2020 ; nous ne l'avons pas.
  Cela change légèrement la distribution des pics journaliers, donc le pic typique.
- **Demande horaire** : heure = (SETTLEMENT_PERIOD − 1) // 2, moyenne des deux demi-heures.
- **Jours invalides** : jours sans exactement 48 demi-heures, soit les 6 jours de changement d'heure
  (fin mars : 46, fin octobre : 50).

## Pic typique (mise à l'échelle globale)

demande normalisée = demande / pic_typique × 0,5 × capacité_totale, avec une seule échelle pour toutes
les données. Un jour typique a donc un pic à 50 % de la capacité, les jours d'hiver sont plus hauts et
les jours d'été plus bas.

Le pic journalier est le maximum de la demande horaire de chaque date, sur toutes les données 2017-2019.

| Méthode | Pic typique |
|---|---|
| Auteurs : `np.mean(daily_peak.mode())`, sans arrondi | **37 378,4 MW** (moyenne de 12 modes) |
| Mode d'une KDE gaussienne (largeur de bande de Scott) | **32 736,7 MW** |
| Écart relatif | **12,42 %** |

Règle : KDE si l'écart est < 2 %, sinon la version des auteurs. L'écart dépasse 2 %, donc
**la version des auteurs est utilisée** (`peak_method="kde"` par défaut, qui retombe sur les auteurs ;
`peak_method="authors"` force ce choix).

Le code des auteurs (`load_demand_data`) a été vérifié : `daily_peak` est le maximum de la demande horaire
par date, puis `typical_max = np.mean(daily_peak.mode())`, **sans arrondi**. Notre calcul est identique.

**Source d'écart possible avec l'article.** Sur des données continues, ce « mode exact » revient presque
à un tirage arbitraire. Presque tous les pics journaliers sont uniques : 12 valeurs apparaissent 2 fois
chacune et s'étalent de 27 638 à 46 776 MW. Le résultat ne dépend pas de l'ordre de tri, car
`pandas.Series.mode()` renvoie *tous* les ex aequo et on en prend la moyenne. Contrairement à
`scipy.stats.mode`, qui renverrait le plus petit, il n'y a donc pas d'arbitraire de tri. En revanche,
le résultat dépend fortement des données exactes : avec le début de 2020 chez les auteurs, les ex aequo,
et donc l'échelle, sont certainement différents. Ce choix fixe l'échelle de toutes les instances
(facteur 0,459).

Nous gardons la version des auteurs pour la réplication. Une **variante KDE** est construite pour un test
de sensibilité : `python uc/instances.py --kde` produit `data/instances_kde/`, et
`python experiments/phase1.py --kde` produit `results/phase1_kde.csv`.

| | Auteurs (référence) | KDE (sensibilité) |
|---|---|---|
| Pic typique | 37 378 MW | 32 737 MW |
| Pic/cap moyen train / val / test | 0,522 / 0,523 / 0,491 | 0,596 / 0,597 / 0,560 |
| Pic/cap hiver / été (train) | 0,620 / 0,434 | 0,708 / 0,495 |
| Écart moyen (q_LPR − LP)/LP, 10 tests | 4,95 × 10⁻⁴ | 5,26 × 10⁻⁴ |

Avec la KDE, la demande est 14 % plus forte partout ; les vérifications de la Phase 1 restent toutes OK.

## Décalage entre train (2017) et test (2019)

Le pic/cap moyen vaut 0,491 sur le test contre 0,522 sur le train, et 0,579 contre 0,620 en hiver.
Pour comprendre cet écart, on compare avec la population : toutes les 360 dates valides de chaque année,
sans bruit.

| Année | Pic/cap moyen | Hiver (DJF) | Printemps (MAM) | Été (JJA) | Automne (SON) |
|---|---|---|---|---|---|
| 2017 | 0,518 | 0,616 | 0,492 | 0,430 | 0,535 |
| 2018 | 0,510 | 0,597 | 0,500 | 0,426 | 0,519 |
| 2019 | 0,494 | 0,576 | 0,476 | 0,413 | 0,515 |

- **Le décalage vient surtout de l'année, pas de l'échantillonnage.** La demande NESO baisse d'année en année :
  le pic journalier brut moyen passe de 37 775 MW en 2017 à 36 079 MW en 2019, soit −4,5 %. L'échantillon de
  test (0,491) correspond à la population 2019 (0,494), et l'hiver du test (0,579) à l'hiver 2019 (0,576).
  Ce décalage découle du découpage par année des auteurs (train 2017, val 2018, test 2019). Il est donc aussi
  présent dans l'article, et le modèle appris devra extrapoler vers une demande un peu plus faible.
- **Répartition par saison des 100 dates de test** : hiver 19, printemps 23, été 33, automne 25, contre environ
  25 attendues par saison. L'été est un peu surreprésenté : P(été ≥ 33) = 0,06 et P(hiver ≤ 19) = 0,11 par la
  loi binomiale. C'est plausible comme hasard d'échantillonnage, avec un effet faible. Train :
  hiver 242, printemps 240, été 260, automne 258. Val : 7, 7, 7, 9.
- Chaque ensemble ne contient que son année (train 2017, val 2018, test 2019).

## Fenêtres de 48 h

- Une instance couvre **48 heures consécutives** (2 jours), de la date d à 00:00 jusqu'à d+1 à 23:00.
  Les temps minimaux de marche et d'arrêt et les rampes de 200_0_1_w sont en heures, donc un pas d'une heure
  est nécessaire.
- **Dates de début valides** : on exclut les jours invalides, le jour qui précède un jour invalide et
  le 31 décembre (le lendemain est dans une autre année). Cela laisse 360 dates par année.
  On vérifie que chaque fenêtre a exactement 48 valeurs.

## Bruit

Le bruit est multiplicatif : instance = base × (1 + ε), où ε est un processus gaussien.

- x = linspace(0, 2, 48) en jours ; noyau exponentiel carré K = exp(−(xᵢ − xⱼ)² / (2 · 0,5²)) + 10⁻¹⁰ I.
- ε = 0,05 · cholesky(K) · z, avec z ~ N(0, I). **0,05 est l'écart-type** du bruit à chaque heure et
  **0,5 jour la longueur de corrélation**.

## Ensembles et graines

| Ensemble | Année | N | Graine |
|---|---|---|---|
| train | 2017 | 1000 | 0 |
| val | 2018 | 30 | 1 |
| test | 2019 | 100 | 2 |

Pour chaque ensemble, `rng = np.random.default_rng(graine)` tire d'abord les N dates (uniformes, avec
remise, parmi les 360 dates valides), puis les N bruits. Les auteurs font autrement : un générateur par
instance, `np.random.RandomState(instance_id + n_g * n_t)`, qui tire la date puis le bruit. Nos instances
suivent donc le même protocole statistique, mais ce ne sont pas les mêmes tirages. La grille du bruit est
la même : `x = linspace(0, n_t / 24, n_t)`.
Fichiers : `data/instances/{train,val,test}.npz`.

## Modèle

- **Tous les générateurs sont EN MARCHE avant l'horizon**, sans historique. Dans
  `u[t] − (1 if t == 0 else u[t−1]) == v[t] − w[t]`, le « 1 » est l'état **avant** l'horizon (u₋₁ = 1),
  pas la décision à t = 0. La décision u₀ est libre. Cette convention est la même dans le LP
  (`uc/uc_fullmodel.py`) et dans `GeneratorPricing`.
  - **Conforme aux auteurs.** Dans `sucpy/problems/uc.py`, `initial_condition = 0` veut dire « en marche avant
    t = 0, sans temps de marche restant à respecter » (≥ 0 : en marche ; < 0 : à l'arrêt). Le membre de droite
    de la contrainte à t = 0 vaut `initial_condition >= 0`, soit 1. Les valeurs `init_status` des données
    (110 générateurs en marche, 90 à l'arrêt) ne sont pas utilisées par les auteurs, qui ne vérifient que
    l'absence de 0.
  - Il n'y a aucun report d'avant l'horizon : les sommes des temps minimaux de marche et d'arrêt ne portent que
    sur t ≥ 0, et les rampes ne s'appliquent qu'à partir de t = 1 (aucune production initiale imposée).
    **Il n'y a pas de coût d'arrêt** (w a un coût nul, chez nous comme chez les auteurs).
  - Conséquence : un générateur peut s'arrêter à t = 0 (w₀ = 1) sans coût. Avec y = 0, chaque sous-problème
    a un coût réduit ≥ 0 (aucun coût négatif dans 200_0_1_w), donc l'optimum est « tout éteint » et
    **q(0) = 0 exactement**. On l'a vérifié : 200 arrêts à t = 0, Σu = 0, Σp = 0.
    `evaluate(np.zeros(48), np.zeros(48))` met bien à zéro y_d **et** y_r, passés séparément.
    Pour comparaison, q(y_d = 0, y_r = 1) = 64 663 > 0.
- Réserve = 10 % de la demande, à chaque heure.
- Formulations du LP et des sous-problèmes inchangées.

## Mesures

- Les temps sont séparés en **temps_y** (temps pour obtenir les duales : 0 pour le démarrage à froid,
  construction + résolution du LP pour la borne LPR) et **temps_eval** (une évaluation de q(y), soit
  200 sous-problèmes MIP).
- `DualPricing` (les 200 modèles) est construit **une seule fois**, hors chronométrage. Pour chaque
  instance, seule la demande change, avec `set_demand`.
- **Gurobi 13.0.3** (licence académique) au lieu de CPLEX : les temps ne sont pas directement comparables
  à ceux de l'article.

## Résultats de la Phase 1 (10 premières instances de test)

Toutes les vérifications passent : duales ≥ 0, q(y_LPR) ≥ objectif LP, q(0) = 0.
Écart relatif moyen (q_LPR − LP)/LP = 4,95 × 10⁻⁴. Temps moyens : LP 2,57 s, evaluate(y_LPR) 0,93 s,
evaluate(0) 0,33 s. Détail dans `results/phase1.csv`.
