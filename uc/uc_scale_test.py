import sys
import time
from pathlib import Path

import numpy as np

# make uc/ importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from uc.data import load_generators
from uc.uc_pricing import GeneratorPricing
from uc.uc_check import check_schedule

#load the 200 generators
gen = load_generators(ROOT / "data" / "generators" / "out" / "200_0_1_w.csv")
n = len(gen["max_out"])
T = 48
print("generators:", n)

#build the 200 pricers once
start = time.perf_counter()
pricers = [GeneratorPricing(gen, g, T) for g in range(n)]
print("build time:", round(time.perf_counter() - start, 3), "s")

#one set of random prices
rng = np.random.default_rng(0)
y_d = rng.uniform(0, 120, T)
y_r = rng.uniform(0, 10, T)

#solve all 200 and check each one, timing the whole loop
fails = []
start = time.perf_counter()
for g in range(n):
    value, schedule = pricers[g].solve(y_d, y_r)
    if not check_schedule(gen, g, schedule):
        fails.append(g)
q_time = time.perf_counter() - start

#results
print("failed generators:", fails)
print("one q(y):", round(q_time, 3), "s")
