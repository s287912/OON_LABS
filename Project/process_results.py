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

with open('results_full/net_fixed.pkl', 'rb') as inp:
    net_fixed = pickle.load(inp)
