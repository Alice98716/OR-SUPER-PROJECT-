#import packages 

import sys
from pathlib import Path

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gurobipy import GRB


# make uc/ importable (ROOT = project folder)
ROOT = Path.cwd()
while not (ROOT / "uc").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from uc.data import load_generators
from uc.uc_bound import DualPricing

#import data

INSTANCE = "10_0_1_w"  # <n_generators>_0_<problem>_w
DATA_DIR = ROOT / "data" / "generators" / "out"

gen = load_generators(DATA_DIR / f"{INSTANCE}.csv")
n_periods = 48                                   
t = np.arange(n_periods)
capacity = gen["max_out"].sum()
shape = 0.8 + 0.2 * np.sin(2 * np.pi * (t - 12) / n_periods)   #between 0.6 and 1.0
demand = 0.5 * capacity * shape                  #peak = 50% of capacity


G = range(gen["n_g"])   # generators
T = range(len(demand))  # hours

print(f"{gen['n_g']} generators, {len(T)} hours")
print(f"demand: {demand.min():.0f}-{demand.max():.0f} MW, total capacity: {gen['max_out'].sum():.0f} MW")
pd.DataFrame({k: v for k, v in gen.items() if k != "n_g"})


#Model NEED TO REWRITE ALL THE MARKDOWNS. This follows the notation of previous notebookss

def build_uc_model(gen, demand, reserve_rate=0.1):
    """Build a Gurobi model for the unit commitment problem."""
    m = gp.Model("uc_fullmodel")
    m.setParam("OutputFlag", 0)  #not show output of solver 

    G = range(len(gen["max_out"]))  # generators
    D= np.asarray(demand, dtype=float)
    T = len(D)
    H = range(T)
    R= reserve_rate*D

    #create variables with LP relaxation 
    u = m.addVars(G, T, vtype=GRB.CONTINUOUS, name="u")  # on/off
    p = m.addVars(G, T, lb=0.0, vtype= GRB.CONTINUOUS, name="p")             # production
    v = m.addVars(G, T, lb =0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="v")  # start-up
    w = m.addVars(G, T, lb=0.0, ub=1.0,vtype=GRB.CONTINUOUS, name="w")  # shut-down

    #parameters needed for constraints and obj
    min_out = np.asarray(gen["min_out"], dtype=float)
    max_out = np.asarray(gen["max_out"], dtype=float)
    nol_cost = np.asarray(gen["nol_cost"], dtype=float)
    mar_cost = np.asarray(gen["mar_cost"], dtype=float)
    st_cost = np.asarray(gen["st_cost"], dtype=float)
    min_up = np.asarray(gen["min_up"], dtype=int)
    min_down = np.asarray(gen["min_down"], dtype=int)
    op_rampup = np.asarray(gen["op_rampup"], dtype=float)
    op_rampdown = np.asarray(gen["op_rampdown"], dtype=float)
    st_rampup = np.asarray(gen["st_rampup"], dtype=float)
    sh_rampdown = np.asarray(gen["sh_rampdown"], dtype=float)

    #add constraints

    #linking constraints (duals: y_d, y_r)
    demand_cstr = m.addConstrs((gp.quicksum(p[g, t] for g in G) >= D[t] for t in H), name="demand")
    reserve_cstr = m.addConstrs((gp.quicksum(max_out[g] * u[g, t] - p[g, t] for g in G) >= R[t] for t in H), name="reserve")

    #same constraints as earlier 
    m.addConstrs((min_out[g] * u[g, t] <= p[g, t] for g in G for t in H), name="out_lo")
    m.addConstrs((p[g, t] <= max_out[g] * u[g, t] for g in G for t in H), name="out_hi")
    m.addConstrs((u[g,t] - (1 if t==0 else u[g, t-1]) == v[g, t] - w[g, t] for g in G for t in H), name = "logic") 
    m.addConstrs((v[g, t] + w[g, t] <= 1 for g in G for t in H), name="switch")
    m.addConstrs((gp.quicksum(v[g, i] for i in range(max(0, t-min_up[g]+1), t+1)) <= u[g, t] for g in G for t in H), name="min_up")
    m.addConstrs((gp.quicksum(w[g, i] for i in range(max(0, t-min_down[g]+1), t+1)) <= 1 - u[g, t] for g in G for t in H), name="min_down")
    m.addConstrs((p[g, t] - p[g, t-1] <= op_rampup[g] * u[g, t-1] + st_rampup[g] * v[g, t] for g in G for t in range(1, T)), name="ramp_up")
    m.addConstrs((p[g, t-1] - p[g, t] <= op_rampdown[g] * u[g, t] + sh_rampdown[g] * w[g, t] for g in G for t in range(1, T)), name="ramp_down")

    #Ojective function
    m.setObjective(gp.quicksum(nol_cost[g] * u[g, t] + mar_cost[g] * p[g, t] + st_cost[g] * v[g, t] for g in G for t in H), GRB.MINIMIZE)

    return m, demand_cstr, reserve_cstr

def solve_uc_model(gen, demand, reserve_rate=0.1): 
    m, demand_cstr, reserve_cstr = build_uc_model(gen, demand, reserve_rate=reserve_rate)
    m.optimize()
    y_d = np.array([demand_cstr[t].Pi for t in range(len(demand))])
    y_r = np.array([reserve_cstr[t].Pi for t in range(len(demand))])
    return m.ObjVal, y_d, y_r 


lp_obj, y_d, y_r = solve_uc_model(gen, demand)
bound = DualPricing(gen, demand, T=len(demand))
q_lpr, _, _ = bound.evaluate(y_d, y_r)
print("LP objective:", lp_obj)
print("q(y_LPR):   ", q_lpr)
print("q >= LP:    ", q_lpr >= lp_obj - 1e-6)
print("min duals:  ", y_d.min(), y_r.min())   # both should be >= 0