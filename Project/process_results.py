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

with open('results_full/STMS_net_shannon_2.pkl', 'rb') as inp:
    net_shannon = pickle.load(inp)
    traffic_matrix_shannon = pickle.load(inp)

#with pd.option_context('display.max_rows', None):
 #   print(net_shannon.route_space.to_string)

df = net_shannon.route_space
print(df)
print(df.loc[df['Ch.9'] == 1])