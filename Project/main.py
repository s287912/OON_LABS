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
if __name__ == '__main__':
    M = 8
    net = elem.Network('resources/full_network.json')
    print(net.weighted_paths)
    net.draw

