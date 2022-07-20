import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import copy
import csv
from core import utils as util
#from core import elements_lab7 as elem
from core import elements as elem
#from core import elements_git as elem2
import math
import time as time
import pickle
with open('results_full/STMS_net_fixed.pkl', 'rb') as inp:
    net_fixed_1 = pickle.load(inp)
    traffic_matrix_fixed_1 = pickle.load(inp)
with open('results_full/STMS_net_fixed.pkl', 'rb') as inp:
    net_fixed_2 = pickle.load(inp)
    traffic_matrix_fixed_2 = pickle.load(inp)
with open('results_full/STMS_net_flex.pkl', 'rb') as inp:
    net_flex_1 = pickle.load(inp)
    traffic_matrix_flex_1 = pickle.load(inp)
with open('results_full/STMS_net_flex_2.pkl', 'rb') as inp:
    net_flex_2 = pickle.load(inp)
    traffic_matrix_flex_2 = pickle.load(inp)
with open('results_full/STMS_net_shannon.pkl', 'rb') as inp:
    net_shannon_1 = pickle.load(inp)
    traffic_matrix_shannon_1 = pickle.load(inp)
with open('results_full/STMS_net_shannon_2.pkl', 'rb') as inp:
    net_shannon_2 = pickle.load(inp)
    traffic_matrix_shannon_2 = pickle.load(inp)

#with pd.option_context('display.max_rows', None):
 #   print(net_shannon.route_space.to_string)

#net_fixed_1.draw
#plt.savefig('results_full/Network_graph.png', dpi = 150)
#plt.show()

