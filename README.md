# OR-SUPER-PROJECT-
IFT 6575 - Final Project Code 

## Pour le volet apprentissage

Les instances (200 générateurs, 48 h) sont dans `data/instances/{train,val,test}.npz`.
Pour les reconstruire : `python uc/instances.py`. Les choix de construction sont décrits dans `notes_hypotheses.md`.

Charger un ensemble :

```python
from uc.instances import load_instances

demand, base_dates = load_instances("train")   # demand : (1000, 48) en MW ; base_dates : dates ISO
```

Évaluer la borne q(y) pour des duales prédites :

```python
import numpy as np
from uc.data import load_generators
from uc.uc_bound import DualPricing

gen = load_generators()
bound = DualPricing(gen, demand[0])      # construit les 200 sous-problèmes une seule fois
for d, y_d, y_r in zip(demand, pred_y_d, pred_y_r):
    bound.set_demand(d)                  # change la demande sans reconstruire
    y_d = np.maximum(y_d, 0)             # 48 valeurs >= 0
    y_r = np.maximum(y_r, 0)             # 48 valeurs >= 0
    q, _, _ = bound.evaluate(y_d, y_r)   # borne inférieure : plus q est grand, mieux c'est
```

Les duales LP de référence pour les 10 premières instances de test sont dans
`results/lpr_duals_test10.npz` (`y_d`, `y_r`, `instance`). La Phase 1 se relance avec `python experiments/phase1.py`.
