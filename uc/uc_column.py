import numpy as np


class ColumnPool:
    def __init__(self, gen):
       
        self.nol_cost = np.array(gen["nol_cost"], dtype=float)
        self.mar_cost = np.array(gen["mar_cost"], dtype=float)
        self.st_cost = np.array(gen["st_cost"], dtype=float)
        self.max_out = np.array(gen["max_out"], dtype=float)

        self.n_g = gen["n_g"]  
        self.T = 48

        self.cost = [[] for g in range(self.n_g)]
        self.P = [[] for g in range(self.n_g)]
        self.S = [[] for g in range(self.n_g)]
        self.keys = [set() for g in range(self.n_g)]  #pour les doublons

    def create_column(self, g, sched):
        p = sched["p"]
        u = sched["u"]
        v = sched["v"]

        cost = self.nol_cost[g] * np.sum(u) + self.mar_cost[g] * np.sum(p) + self.st_cost[g] * np.sum(v)
        s = self.max_out[g] * u - p

        return cost, p, s

    def add_column(self, g, sched):
        cost, p, s = self.create_column(g, sched)
        key = tuple(np.round(p, 6)) + tuple(sched["u"])

        if key not in self.keys[g]:
            self.P[g].append(p)
            self.S[g].append(s)
            self.cost[g].append(cost)
            self.keys[g].add(key)

    def add_all(self, schedules):
        for g in range(self.n_g):
            self.add_column(g, schedules[g])

    def value(self, g, k, y_d, y_r):
        p = self.P[g][k]
        s = self.S[g][k]
        cost = self.cost[g][k]

        return cost - np.dot(y_d, p) - np.dot(y_r, s)

    def model_value(self, D, R, y_d, y_r):
        m = np.dot(D, y_d) + np.dot(R, y_r)
        for g in range(self.n_g):
            Pg = np.array(self.P[g])
            Sg = np.array(self.S[g])
            costg = np.array(self.cost[g])
            valeurs = costg - np.dot(Pg, y_d) - np.dot(Sg, y_r)
            m += np.min(valeurs)

        return m
