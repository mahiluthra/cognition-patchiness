# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:24:33 2019

@author: mahi
"""

import model_patchcog as model
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import time
from matplotlib import colors

start = time.time()

if __name__ == "__main__":
    trial = sys.argv[1]

#self, num, start_energy, eat_energy, tire_energy, reproduction_energy, cognition_energy, width, height, cognition

#cog_type: 0: no cognition
#cog_type: 1: only plants
#cog_type: 2: plans and animals

print(trial)

sim = model.modelSim(cog_fixed = False, introduce_time =  300, disp_rate = 0, dist = 1, det = 0, skip_300 = False, collect_cog_dist = True, evolve_disp = False) # create world
for i in range(2000): # run simulations
    sim.step()

model_hist = sim.history
model_hist.to_csv(r"data/exp1_" + str(trial) + "_model_hist.csv", sep = ",", header= "True")

dist_hist = sim.cog_dist_dist
det_hist = sim.cog_dist_det
dist_hist.to_csv(r"data/exp1_" + str(trial) + "_dist_hist.csv", sep = ",", header= "True")
det_hist.to_csv(r"data/exp1_" + str(trial) + "_det_hist.csv", sep = ",", header= "True")


end = time.time()
elapsed = end - start
print(elapsed)