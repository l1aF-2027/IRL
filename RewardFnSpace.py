from pulp import *
import Config as C
from Reward import Reward
from more_itertools import unzip
import numpy as np
import logging
logging.basicConfig(filename='irl.log', level=logging.INFO)

class RewardFnSpace () : 

    def __init__ (self, rewardBases) : 
        self.rewardBases = rewardBases
        self.rng = np.random.RandomState(C.SEED)
        self._initializeLP ()

    def _initializeLP (self) : 
        self.lp = LpProblem('IRL', LpMaximize)
        self.y1s, self.y2s, self.alphas, self.bs = [], [], [], []
        for i, _ in enumerate(self.rewardBases):  
            y1 = LpVariable(f'y1_{i}')
            y2 = LpVariable(f'y2_{i}')
            self.y1s.append(y1)
            self.y2s.append(y2)
            self.alphas.append(1 - (y1 - y2))
        for y1 in self.y1s : 
            self.lp += y1 >= 0 
        for y2 in self.y2s : 
            self.lp += y2 >= 0
        for alpha in self.alphas : 
            self.lp += alpha >= 0
            self.lp += alpha <= 1
        self.l1Term = lpSum([-C.L1_NORM_GAMMA * (y1 + y2) 
                             for y1, y2 in zip(self.y1s, self.y2s)])
        self.lp += self.l1Term
        self.coeffs = [self.rng.rand() for _ in self.rewardBases]

    def _estimatedValueExpressions (self, stateValuesForBases) :
        svfb = np.array(stateValuesForBases)
        alphas = np.array(self.alphas)
        estimates = (svfb.T * alphas).sum(axis=1).tolist()
        return estimates

    def _setCoeffs (self) : 
        self.coeffs = [value(a) for a in self.alphas]

    def current (self) : 
        pairs = list(zip(self.coeffs, self.rewardBases))
        fn = lambda s : sum([c * rfn(s) for c, rfn in pairs])
        ranges = [rfn.reward_range for rfn in self.rewardBases]
        mins, maxs = list(map(list, unzip(ranges)))
        rMin = min(c * m for c, m in zip(self.coeffs, mins))
        rMax = max(c * M for c, M in zip(self.coeffs, maxs))
        return Reward(fn, (rMin, rMax))

    def refine (self, expertValues, inferiorValues) :
        n = len(self.bs)
        expertEstimates = self._estimatedValueExpressions(expertValues)
        inferiorEstimates = self._estimatedValueExpressions(inferiorValues)
        for i, (exp, inf) in enumerate(
                zip(expertEstimates, inferiorEstimates)) : 
            b = LpVariable(f'b_{n + i}')
            self.lp += b <= 2 * (exp - inf)
            self.lp += b <=     (exp - inf)
            self.bs.append(b)
        self.lp += lpSum(self.bs) + self.l1Term
        self.lp.solve()
        self._setCoeffs()