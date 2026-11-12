import numpy as np 
import time
from uc.uc_pricing import GeneratorPricing


class DualPricing: 

    def __init__(self, gen, demand, reserve_rate=0.1, T=48):
        self.gen=gen
        self.demand = demand
        self.reserve_rate = reserve_rate
        self.T = T 
        self.D = np.asarray(demand, dtype = float)
        self.R = reserve_rate * self.D

        self.max_out = np.asarray(gen["max_out"])
        self.number_of_generators = len(self.max_out)
        self.pricers = []
        for g in range(self.number_of_generators):
            pr = GeneratorPricing(gen, g, T)
            self.pricers.append(pr)

    def evaluate(self, y_d, y_r): 
        y_d = np.asarray(y_d, dtype = float)
        y_r = np.asarray(y_r, dtype = float)

        q = sum(y_d*self.D + y_r*self.R)

        supply = np.zeros(self.T)
        spare = np.zeros(self.T)
        schedules = []
        for g in range(self.number_of_generators):
                value, sched = self.pricers[g].solve(y_d, y_r)
                q += value
                supply += sched["p"] 
                spare +=self.max_out[g] * sched["u"] - sched["p"] 
                schedules.append(sched)
        g_d =  self.D - supply 
        g_r = self.R - spare
        return q, schedules, (g_d, g_r)