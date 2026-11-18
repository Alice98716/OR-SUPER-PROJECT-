"""Phase 1 : borne froide q(0), relaxation LP et borne q(y_LPR) sur les 10 premières
instances de test (2019, 200 générateurs, 48 h).

- DualPricing (200 sous-problèmes) est construit UNE fois, hors chronométrage ;
  pour chaque instance on change seulement la demande avec set_demand.
- temps_y    : temps pour obtenir les duales y (0 pour froid, construction + résolution LP pour lpr)
- temps_eval : temps d'une évaluation de la borne q(y) (0 pour lpr_objectif)

Sorties : results/phase1.csv, results/lpr_duals_test10.npz
Lancer : python experiments/phase1.py         (après python uc/instances.py)
         python experiments/phase1.py --kde   (variante de sensibilité, après python uc/instances.py --kde)
                                              -> results/phase1_kde.csv, results/lpr_duals_test10_kde.npz
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from uc.data import load_generators
from uc.instances import INSTANCE_DIR, INSTANCE_DIR_KDE, load_instances
from uc.uc_bound import DualPricing
from uc.uc_fullmodel import solve_uc_model

N_INSTANCES = 10
RESULTS = ROOT / "results"
USE_KDE = "--kde" in sys.argv
SUFFIX = "_kde" if USE_KDE else ""

gen = load_generators()
demand_all, dates_all = load_instances("test", INSTANCE_DIR_KDE if USE_KDE else INSTANCE_DIR)
demands, dates = demand_all[:N_INSTANCES], dates_all[:N_INSTANCES]
T = demands.shape[1]
print(f"{gen['n_g']} générateurs, {N_INSTANCES} instances de test, T = {T}")

# Construction unique des 200 sous-problèmes (non chronométrée).
t0 = time.perf_counter()
bound = DualPricing(gen, demands[0], T=T)
print(f"construction de DualPricing (une fois) : {time.perf_counter() - t0:.2f} s\n")

rows, duals_d, duals_r, names, checks = [], [], [], [], []
zeros = np.zeros(T)
for i in range(N_INSTANCES):
    name = f"test_{i:03d}_{dates[i]}"
    demand = demands[i]
    bound.set_demand(demand)

    # Démarrage à froid : y = 0.
    t0 = time.perf_counter()
    q0, _, _ = bound.evaluate(zeros, zeros)
    t_cold = time.perf_counter() - t0

    # Relaxation LP : objectif et duales (construction + résolution).
    t0 = time.perf_counter()
    lp_obj, y_d, y_r = solve_uc_model(gen, demand)
    t_lp = time.perf_counter() - t0

    # Borne lagrangienne aux duales LP.
    t0 = time.perf_counter()
    q_lpr, _, _ = bound.evaluate(y_d, y_r)
    t_eval = time.perf_counter() - t0

    rows += [
        ("froid", name, q0, 0.0, t_cold),
        ("lpr_borne", name, q_lpr, t_lp, t_eval),
        ("lpr_objectif", name, lp_obj, t_lp, 0.0),
    ]
    duals_d.append(y_d)
    duals_r.append(y_r)
    names.append(name)

    # Vérifications.
    ok_duals = y_d.min() >= -1e-9 and y_r.min() >= -1e-9
    ok_bound = q_lpr >= lp_obj - 1e-6 * abs(lp_obj)
    ok_cold = abs(q0) <= 1e-6
    checks.append((ok_duals, ok_bound, ok_cold))
    tag = lambda ok: "OK" if ok else "ÉCHEC"
    print(f"{name}  LP {lp_obj:14.2f} ({t_lp:4.2f}s)  q_LPR {q_lpr:14.2f} ({t_eval:4.2f}s)  "
          f"q0 {q0:9.2e} ({t_cold:4.2f}s)  | duales>=0 {tag(ok_duals)}  "
          f"q_LPR>=LP {tag(ok_bound)}  q0=0 {tag(ok_cold)}")

RESULTS.mkdir(exist_ok=True)
df = pd.DataFrame(rows, columns=["methode", "instance", "valeur", "temps_y", "temps_eval"])
df.to_csv(RESULTS / f"phase1{SUFFIX}.csv", index=False)
np.savez(RESULTS / f"lpr_duals_test10{SUFFIX}.npz", y_d=np.array(duals_d), y_r=np.array(duals_r),
         instance=np.array(names))

# Résumé.
lp = df[df.methode == "lpr_objectif"].valeur.to_numpy()
qb = df[df.methode == "lpr_borne"].valeur.to_numpy()
print("\nRésumé")
print(f"  écart relatif moyen (q_LPR - LP)/LP : {np.mean((qb - lp) / lp):.3e}"
      f"  (min {np.min((qb - lp) / lp):.3e}, max {np.max((qb - lp) / lp):.3e})")
print(f"  temps moyen LP (construction + résolution) : {df[df.methode == 'lpr_borne'].temps_y.mean():.2f} s")
print(f"  temps moyen evaluate(y_LPR)                : {df[df.methode == 'lpr_borne'].temps_eval.mean():.2f} s")
print(f"  temps moyen evaluate(0)                    : {df[df.methode == 'froid'].temps_eval.mean():.2f} s")
print(f"  écrit : results/phase1{SUFFIX}.csv, results/lpr_duals_test10{SUFFIX}.npz")
all_ok = all(all(c) for c in checks)
print("Phase 1 TERMINÉE : toutes les vérifications OK" if all_ok
      else f"Phase 1 NON terminée : {sum(not all(c) for c in checks)} instance(s) en échec")
