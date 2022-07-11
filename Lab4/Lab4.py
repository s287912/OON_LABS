import random

from  import elements as elem
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import c

net = elem.Network('../Lab5/resources/nodes.json')
#print(net.weighted_paths)
#print(net.find_best_snr('A','D'))
#print(net.find_best_latency('A','D'))

conn = elem.Connection('A','D',1)
conn2 = elem.Connection('D','F',1)
net.stream([conn, conn2],'snr')
print(conn.snr)
print(conn2.snr)
net.lines['AB'].state = 'occupied'

connections = []
for i in range(0,100):
    nodes = list(net.nodes.keys())
    node1 = random.choice(nodes)
    nodes.remove(node1)
    node2 = random.choice(nodes)
#    print("From ", node1, " to ", node2)
    conn = elem.Connection(node1,node2,1)
    connections.append(conn)

net.stream(connections,'snr')
net.stream(connections,'latency')

for conn in connections:
    print(conn.input,'->',conn.output,'best snr:', conn.snr, 'best latency:', conn.latency)

print(net.weighted_paths)
#net.draw