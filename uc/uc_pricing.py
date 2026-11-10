import gurobipy as gp 
from gurobipy import GRB

import numpy as np 

class GeneratorPricing: 
    def __init__(self, gen, g, T=48):
        self.gen = gen
        self.g = g
        self.T = T
        hours = range(T)
        
        
        self.max_out = gen["max_out"][g]
        self.nol_cost = gen["nol_cost"][g]
        self.mar_cost = gen["mar_cost"][g]
        min_out = gen["min_out"][g]
        st_cost = gen["st_cost"][g]
        min_up = int(gen["min_up"][g])
        min_down = int(gen["min_down"][g])
        op_rampup = gen["op_rampup"][g]
        op_rampdown = gen["op_rampdown"][g]
        st_rampup = gen["st_rampup"][g]
        sh_rampdown = gen["sh_rampdown"][g]


        self.m = gp.Model("GeneratorPricing")
        self.m.setParam('OutputFlag', 0) 
        self.m.setParam('IntFeasTol', 1e-9)

        self.m.ModelSense=GRB.MINIMIZE # Suppress Gurobi output

        self.u = self.m.addVars(T, vtype=GRB.BINARY, name="u")
        self.p = self.m.addVars(T, vtype=GRB.CONTINUOUS, name="p")
        self.v = self.m.addVars(T, vtype=GRB.CONTINUOUS, name="v")
        self.w = self.m.addVars(T, vtype=GRB.CONTINUOUS, name="w")
        u, p, v, w = self.u, self.p, self.v, self.w
        
        self.m.addConstrs((min_out * u[t] <= p[t] for t in hours), name="out_lo")
        self.m.addConstrs((p[t] <= self.max_out * u[t] for t in hours), name="out_hi")


        self.m.addConstrs((u[t] - (1 if t == 0 else u[t - 1]) == v[t] - w[t] for t in hours))
        self.m.addConstrs((v[t] + w[t] <= 1 for t in hours))
        self.m.addConstrs( (gp.quicksum(v[i] for i in range(max(0, t - min_up + 1), t + 1)) <= u[t]for t in hours))
        self.m.addConstrs((gp.quicksum(w[i] for i in range(max(0, t - min_down + 1), t + 1)) <= 1 - u[t] for t in hours))
        self.m.addConstrs((p[t] - p[t - 1] <= op_rampup * u[t - 1] + st_rampup * v[t]for t in range(1, T)))
        self.m.addConstrs((p[t - 1] - p[t] <= op_rampdown * u[t] + sh_rampdown * w[t] for t in range(1, T)))


        for t in hours:
            v[t].Obj = st_cost
            w[t].Obj = 0.0


    def solve(self, y_d, y_r):
        y_d = np.asarray(y_d)
        y_r = np.asarray(y_r)
        hours = range(self.T)


        coef_u = self.nol_cost - y_r * self.max_out
        self.m.setAttr("Obj", [self.u[t] for t in hours], coef_u.tolist())
        coef_p = self.mar_cost - y_d + y_r
        self.m.setAttr("Obj", [self.p[t] for t in hours], coef_p.tolist())

        self.m.optimize()
        schedule = {
            "p": np.array([self.p[t].X for t in hours]),
            "u": np.rint([self.u[t].X for t in hours]).astype(int),
            "v": np.rint([self.v[t].X for t in hours]).astype(int),
            "w": np.rint([self.w[t].X for t in hours]).astype(int),
        }
        return self.m.ObjVal, schedule
