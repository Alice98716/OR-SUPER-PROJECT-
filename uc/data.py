"""Load unit commitment data.

`load_generators` is adapted from `_create_generator_list_frangioni` in
sucpy/problems/uc.py (https://github.com/nsugishita/ml_to_warmstart_cg).
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Repository root, so the default path works from any working directory.
ROOT = Path(__file__).resolve().parent.parent
GENERATOR_FILE = ROOT / "data" / "generators" / "out" / "200_0_1_w.csv"


def load_generators(filepath=GENERATOR_FILE):
    """Load generator data and convert it to the format used in the model.

    Returns a dict of numpy arrays (one entry per generator) plus
    `n_g`, the number of generators.
    """
    df = pd.read_csv(filepath)

    # Reindex generators from 0 to len(df) - 1.
    df["generator_index"] = np.arange(len(df))

    # Convert generator data to one which are compatible to the experiment.
    df = df.set_index("generator_index")
    df_out = pd.DataFrame(index=df.index)
    min_out = df["min_out"]
    max_out = df["max_out"]
    min_cost = (
        df["constant_cost"]
        + df["linear_cost"] * min_out
        + df["quadratic_cost"] * (min_out**2)
    )
    max_cost = (
        df["constant_cost"]
        + df["linear_cost"] * max_out
        + df["quadratic_cost"] * (max_out**2)
    )
    df_out["mar_cost"] = (max_cost - min_cost) / (max_out - min_out)
    df_out["nol_cost"] = min_cost - df_out["mar_cost"] * min_out
    df_out["original_cost_constant"] = df["constant_cost"]
    df_out["original_cost_linear"] = df["linear_cost"]
    df_out["original_cost_quadratic"] = df["quadratic_cost"]

    # It seems the description of the data file is not correct,
    # since `hot_and_fuel_cost` seems to be 0 anywhere.
    # Instead, reading values in the following way seems more realistic.
    # - fixed_cost -> cool_and_fuel_cost
    # - succ -> hot_and_fuel_cost
    # - p0 -> fixed_cost
    df_out["st_cost"] = df["succ"] * 12.0 + df["p0"]
    df_out["min_out"] = df["min_out"]
    df_out["max_out"] = df["max_out"]
    df_out["op_rampup"] = df["ramp_up"]
    df_out["op_rampdown"] = df["ramp_down"]
    # Allow them to ramp up/down at least min_out otherwise
    # they can't switch on/off.
    df_out["st_rampup"] = np.maximum(
        df["ramp_up"].values, df["min_out"].values
    )
    df_out["sh_rampdown"] = np.maximum(
        df["ramp_down"].values, df["min_out"].values
    )
    df_out["min_up"] = df["min_up"]
    df_out["min_down"] = df["min_down"]
    df_out["notification"] = (df["min_up"] * 0.7).astype(int).clip(0)
    if np.any(df["init_status"] == 0):
        raise ValueError("some init_status is 0.")

    df_out["initial_condition"] = 0

    ret = {}
    for col in df_out.columns:
        ret[col] = df_out[col].values
    ret["n_g"] = len(df_out)
    return ret
