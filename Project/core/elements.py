import copy
import json
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import c
from core import parameters as param
from core import science_utils as science
import random as rand

class SignalInformation(object):
    def __init__(self, power, path):
        self._signal_power = float(power)
        self._noise_power = float(0.0)
        self._latency = float(0.0)
        self._path = path
    @property
    def signal_power(self):
        return self._signal_power
    @signal_power.setter
    def signal_power(self, power):
        self._signal_power = power
    @property
    def path(self):
        return self._path
    @path.setter
    def path(self, path):
        self._path = path
    @property
    def noise_power(self):
        return self._noise_power
    @noise_power.setter
    def noise_power(self,noise):
        self._noise_power = noise
    @property
    def latency(self):
        return self._latency
    @latency.setter
    def latency(self, latency):
        self._latency = latency
    def add_noise(self, noise):
        self.noise_power += noise
    def add_latency(self, latency):
        self.latency += latency
    def next(self):
        self.path = self.path[1:]

class Lightpath(SignalInformation):
    def __init__(self, signal_power, path, channel):
        SignalInformation.__init__(self, signal_power, path)
        self._channel = channel
        self._Rs = param.Rs
        self._df = param.df
    @property
    def channel(self):
        return self._channel
    @property
    def Rs(self):
        return self._Rs
    @property
    def df(self):
        return self._df


class Node(object):
    def __init__(self, node_dict):
        self._label = node_dict['label']
        self._position = node_dict['position']
        self._connected_nodes = node_dict['connected_nodes']
        self._successive = {}
        self._switching_matrix = None
        self._transceiver = ''
    @property
    def label(self):
        return self._label
    @property
    def position(self):
        return self._position
    @property
    def connected_nodes(self):
        return self._connected_nodes
    @property
    def successive(self):
        return self._successive
    @successive.setter
    def successive(self, successive):
        self._successive = successive
    @property
    def switching_matrix(self):
        return self._switching_matrix
    @switching_matrix.setter
    def switching_matrix(self, switching_matrix):
        self._switching_matrix = switching_matrix
    @property
    def transceiver(self):
        return self._transceiver
    @transceiver.setter
    def transceiver(self,transceiver):
        self._transceiver = transceiver
    def propagate(self, lightpath, prev_node):
        path = lightpath.path
        if len(path) > 1:
            line_label = path[:2]
            if type(lightpath) is Lightpath:
                if prev_node is not None:
                    channels = self.switching_matrix[prev_node][line_label[1]]
                    channels[lightpath.channel] = 0
                    if lightpath.channel != (param.channels - 1):
                        channels[lightpath.channel + 1] = 0
                    if lightpath.channel != 0:
                        channels[lightpath.channel - 1] = 0
            line = self.successive[line_label]
            lightpath.signal_power = line.optimized_launch_power()
            lightpath.next()
            lightpath = line.propagate(lightpath)
        return lightpath

class Line(object):
    def __init__(self, line_dict):
        self._label = line_dict['label']
        self._length = line_dict['length']
        self._successive = {}
        self._state = [1] * param.channels
        self._n_amplifier = 0
        self._n_span = 0
        self._gain = param.G # linear
        self._noise_figure = param.NF #linear
        self._alfa = param.alfa_linear
        self._beta2 = param.beta2
        self._gamma = param.gamma
    @property
    def label(self):
        return self._label
    @property
    def length(self):
        return self._length
    @property
    def successive(self):
        return self._successive
    @successive.setter
    def successive(self, successive):
        self._successive = successive
    @property
    def state(self):
        return self._state
    @state.setter
    def state(self, state):
        self._state = state
    @property
    def n_amplifier(self):
        return self._n_amplifier
    @n_amplifier.setter
    def n_amplifier(self, n_amplifier):
        self._n_amplifier = n_amplifier
        self._n_span = n_amplifier - 1
    @property
    def n_span(self):
        return self._n_span
    @property
    def gain(self):
        return self._gain
    @property
    def noise_figure(self):
        return self._noise_figure
    @property
    def alfa(self):
        return self._alfa
    @property
    def beta2(self):
        return self._beta2
    @property
    def gamma(self):
        return self._gamma
    def latency_generation(self):
        latency = self.length / (c * 2 / 3)
        return latency
    def noise_generation(self, signal_power):
        #noise = 1e-9 * signal_power * self.length #old one
        noise = self.nli_generation(signal_power) + self.ase_generation()
        return noise
    def propagate(self, lightpath):
        latency = self.latency_generation()
        lightpath.add_latency(latency)
        signal_power = lightpath.signal_power
        noise = self.noise_generation(signal_power)
        lightpath.add_noise(noise)
        node = self.successive[lightpath.path[0]]

        if type(lightpath) == Lightpath:
            self.state[lightpath.channel] = 0
            lightpath = node.propagate(lightpath, self.label[0])
        else:
            lightpath = node.propagate(lightpath, None)
        return lightpath
    def probe(self, signal_information):
        latency = self.latency_generation()
        signal_information.add_latency(latency)
        signal_power = signal_information.signal_power
        noise = self.noise_generation(signal_power)
        signal_information.add_noise(noise)
        node = self.successive[signal_information.path[0]]
        signal_information = node.propagate(signal_information)
        return signal_information
    def ase_generation(self):
        self.n_amplifier = (math.ceil(self.length / 80e3) - 1) + 2
        ASE = self.n_amplifier * (param.h * param.f * param.Bn * self.noise_figure * (self.gain - 1))
        return ASE
    def eta_nli(self):
        Leff = 1 / (2 * self.alfa)
        eta_nli = 16 / (27 * np.pi) * np.log((np.pi ** 2) / 2 * self.beta2 * (param.Rs ** 2) / self.alfa \
                * (param.channels ** (2 * param.Rs / param.df))) \
               * (self.alfa / self.beta2 * ((self.gamma ** 2) * (Leff ** 2) / (param.Rs ** 3)))
        return eta_nli
    def nli_generation(self, signal_power):
        eta_nli = self.eta_nli()
        NLI = (signal_power ** 3) * eta_nli * self.n_span * param.Bn
        return NLI
    def optimized_launch_power(self):
        Popt = (self.ase_generation() / (2 * param.Bn * self.n_span * self.eta_nli())) ** (1/3)
        return Popt

class Network(object):
    def __init__(self, json_path):
        node_json = json.load(open(json_path,'r'))
        self._nodes = {}
        self._lines = {}
        self._weighted_paths = pd.DataFrame()
        self._route_space = pd.DataFrame()
        self._switching_matrix = {}
        self._blocking_count = 0
        for node_label in node_json:
            # create node instance
            node_dict = node_json[node_label]
            node_dict['label'] = node_label
            node = Node(node_dict)
            if 'transceiver' in node_json[node_label].keys():
                node.transceiver = node_json[node_label]['transceiver']
            else:
                node.transceiver = 'fixed_rate'
            self._nodes[node_label] = node
            #create line instances
            for connected_node_label in node_dict['connected_nodes']:
                line_dict = {}
                line_label = node_label + connected_node_label
                line_dict['label'] = line_label
                node_position = np.array(node_json[node_label]['position'])
                connected_node_position = np.array(node_json[connected_node_label]['position'])
                line_dict['length'] = np.sqrt(np.sum((node_position-connected_node_position)**2))
                line = Line(line_dict)
                self._lines[line_label] = line
            self._switching_matrix[node_label] = node_dict['switching_matrix']
            #print("switching_matrix", self._switching_matrix_dict)
            #exit()
        #create the weight
        self.connect()
        node_labels = self.nodes.keys()
        pairs = []
        for label1 in node_labels:
            for label2 in node_labels:
                if label1 != label2:
                    pairs.append(label1+label2)
        paths = []
        latencies = []
        noises = []
        snrs = []
        path_state = []
        for pair in pairs:
            for path in self.find_paths(pair[0], pair[1]):
                path_string = ''
                for node in path:
                    path_string += node + '->'
                paths.append(path_string[:-2])

                signal_information = SignalInformation(1e-3, path)
                signal_information = self.probe(signal_information)
                latencies.append(signal_information.latency)
                noises.append(signal_information.noise_power)
                snrs.append(10 * np.log10(signal_information.signal_power / signal_information.noise_power))
                path_state.append(1)
        self._weighted_paths['path'] = paths
        self._weighted_paths['latency'] = latencies
        self._weighted_paths['noises'] = noises
        self._weighted_paths['snr'] = snrs
        for nch in range(0, param.channels):
            self._route_space['Ch.'+str(nch)] = path_state
    @property
    def weighted_paths(self):
        return self._weighted_paths
    @property
    def nodes(self):
        return self._nodes
    @property
    def lines(self):
        return self._lines
    @property
    def route_space(self):
        return self._route_space
    @property
    def switching_matrix(self):
        return self._switching_matrix
    @switching_matrix.setter
    def switching_matrix(self, switching_matrix):
        self._switching_matrix = switching_matrix
    @property
    def blocking_count(self):
        return self._blocking_count
    @blocking_count.setter
    def blocking_count(self, blocking_count):
        self._blocking_count = blocking_count
    @property
    def draw(self):
        nodes = self.nodes
        plt.figure(figsize=(12, 8))
        xlist = []
        ylist = []
        for node_label in nodes:
            n0 = nodes[node_label]
            x0 = n0.position[0]
            y0 = n0.position[1]
            xlist.append(x0)
            ylist.append(y0)
            plt.plot(x0,y0,'ro',markersize=10)
            plt.text(x0,y0+20000,node_label)
            for connected_node_label in n0.connected_nodes:
                n1 = nodes[connected_node_label]
                x1 = n1.position[0]
                y1 = n1.position[1]
                plt.plot([x0,x1], [y0,y1], 'b', linewidth=1)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

        plt.grid(color='k', linestyle='--', linewidth=0.5)
        plt.xticks(np.linspace(min(xlist), max(xlist), 10))
        plt.yticks(np.linspace(min(ylist), max(ylist), 10))
        plt.title('Network 287912 - Simone Cascianelli')

        #plt.show()
    def find_paths(self, label1, label2):
        cross_nodes = [key for key in self.nodes.keys() if ((key != label1) & (key != label2))]
        cross_lines = self.lines.keys()
        inner_paths = {'0': label1}
        for i in range(len(cross_nodes) + 1):
            inner_paths[str(i+1)] = []
            for inner_path in inner_paths[str(i)]:
                inner_paths[str(i+1)] += [inner_path + cross_node for cross_node in cross_nodes if ((inner_path[-1]+cross_node in cross_lines) & (cross_node not in inner_path))]
            paths = []
        for i in range(len(cross_nodes)+1):
            for path in inner_paths[str(i)]:
                if path[-1] + label2 in cross_lines:
                    paths.append(path + label2)
        return paths
    def connect(self):
        nodes_dict = self.nodes
        lines_dict = self.lines
        for node_label in nodes_dict:
            node = nodes_dict[node_label]
            node.switching_matrix = copy.deepcopy(self.switching_matrix[node_label])
            for connected_node in node.connected_nodes:
                line_label = node_label + connected_node
                line = lines_dict[line_label]
                line.successive[connected_node] = nodes_dict[connected_node]
                node.successive[line_label] = lines_dict[line_label]
    def propagate(self, lightpath):
        path = lightpath.path
        start_node = self.nodes[path[0]]
        propagated_signal_information = start_node.propagate(lightpath, None)
        return propagated_signal_information
    def probe(self, signal_information):
        path = signal_information.path
        start_node = self.nodes[path[0]]
        propagated_signal_information = start_node.propagate(signal_information,None)
        return propagated_signal_information
    def find_best_snr(self, input_label, output_label):
        paths_df = self.weighted_paths
        paths_df = paths_df.loc[(paths_df['path'].str[0] == input_label) & (paths_df['path'].str[-1] == output_label)]
        paths_df = paths_df.sort_values(['snr'], inplace=False, ascending=False)
        #print(paths_df)
        #exit()
        best_path = ''
        channel = None
        for path in paths_df['path'].str.replace('->', ''):
            channel = self.find_channel(path)
            if channel != None:
                #print(path,channel)
                best_path = path
                break
        return best_path, channel
    def find_best_latency(self, input_label, output_label):
        paths_df = self.weighted_paths
        paths_df = paths_df.loc[(paths_df['path'].str[0] == input_label) & (paths_df['path'].str[-1] == output_label)]
        paths_df = paths_df.sort_values(['latency'], inplace=False, ascending=True)
        #print(paths_df)
        #exit()
        best_path = ''
        for path in paths_df['path'].str.replace('->', ''):
            channel = self.find_channel(path)
            if channel != None:
                #print(path,channel)
                best_path = path
                break
        return best_path, channel
    def stream(self, connections, label = 'latency'):
        for connection in connections:
            best_path = ''
            if label == 'snr':
                best_path, channel = self.find_best_snr(connection.input, connection.output)
            else:
                best_path, channel = self.find_best_latency(connection.input, connection.output)
            if (best_path != '') & (channel != None):
                lightpath = Lightpath(connection.signal_power, best_path, channel)
                bit_rate = self.calculate_bit_rate(lightpath, self.nodes[best_path[0]].transceiver)
                if bit_rate == 0:
                    connection.snr = 0
                    connection.latency = 0
                    connection.bit_rate = 0
                else:
                    self.propagate(lightpath)
                    self.update_route_space()
                    connection.bit_rate = bit_rate
                    #print("connect:",connection.input,"->",connection.output, "of path:",best_path,"using channel:", channel)
                    if label == 'snr':
                        connection.snr = 10 * np.log10(lightpath.signal_power / lightpath.noise_power)
                    else:
                        connection.latency = lightpath.latency
            else:
                #df = self.route_space
                #df = df.loc[(self.weighted_paths['path'] == best_path.replace('', '->')[2:-2])]
                #print("Not possible to connect:",connection.input,"->",connection.output)
                #print("Occupation of", best_path)
                #print(df)
                connection.snr = 0.0
                connection.latency = 0.0
                connection.bit_rate = 0
        #self.restore_network()
        #print(self.switching_matrix)
        #exit()
    def find_channel(self, path):
        channel = None
        for ch in range(0, param.channels):
            df = self.route_space
            df = df.loc[(self.weighted_paths['path'] == path.replace('','->')[2:-2])]
            if df['Ch.'+str(ch)].values == 1:
                channel = ch
                break
        return channel
    def update_route_space(self):
        df = self.weighted_paths['path'].str.replace('->','')
        #print(df)
        for path in df:
            #print(path)
            line = self.lines[path[0]+path[1]]
            occupancy = line.state
            node1 = 1
            for node2 in range(2, len(path)):
                line = self.lines[path[node1]+path[node2]]
                occupancy = np.multiply(occupancy, line.state)
                occupancy = np.multiply(self.nodes[path[node1]].switching_matrix[path[node1-1]][path[node2]], occupancy)
                node1 = node2
            #print(occupancy)
            idx = self.weighted_paths.loc[df == path].index.values[0]
            self.route_space.iloc[idx] = occupancy
        return None
    def restore_network(self):
        path_state = [1]*self.weighted_paths.shape[0]
        #print(path_state)
        for nch in range(0, param.channels):
            self._route_space['Ch.'+str(nch)] = path_state
        nodes_dict = self.nodes
        lines_dict = self.lines
        for node_label in nodes_dict:
            nodes_dict[node_label].switching_matrix = copy.deepcopy(self.switching_matrix[node_label])
        for line_label in lines_dict:
            lines_dict[line_label].state = np.ones(param.channels, np.int8)
        self.update_route_space()
        self.blocking_count = 0

    def calculate_bit_rate(self, lightpath, strategy):
        path = lightpath.path
        bit_rate = 0
        gsnr = self.weighted_paths[self.weighted_paths['path'] == path.replace('', '->')[2:-2]]['snr'].values[0]
        #print(gsnr)
        gsnr = float(10 ** (gsnr / 10))

        if strategy == 'fixed_rate':
            bit_rate = science.bit_rate_fixed(gsnr, lightpath.Rs)
        elif strategy == 'flex_rate':
            bit_rate = science.bit_rate_flex(gsnr, lightpath.Rs)
        elif strategy == 'shannon':
            bit_rate = science.bit_rate_shannon(gsnr, lightpath.Rs)

        #print(gsnr, bit_rate)
        return bit_rate
    def traffic_matrix_request(self, traffic_matrix, connections, signal_power, pairs):
        nodes_full = list(self.nodes.keys())
        while self.traffic_matrix_free(traffic_matrix):
            pair = rand.choice(pairs)
            input_node = pair[0]
            output_node = pair[1]
            if traffic_matrix[input_node][output_node] != 0 and traffic_matrix[input_node][output_node] != math.inf:
                break
        # if is necessary has if matrix emptied need to skip the stream
        if self.traffic_matrix_free(traffic_matrix):
            #print(input_node, output_node)
            connection = Connection(input_node, output_node, signal_power)
            current_connections = [connection]
            self.stream(current_connections, 'snr')
            connections.append(connection)
            if connection.snr != 0:
                if connection.bit_rate >= traffic_matrix[input_node][output_node]:
                    traffic_matrix[input_node][output_node] = 0
                    return 1
                else:
                    traffic_matrix[input_node][output_node] -= connection.bit_rate
                    return 0
            else:
                self.blocking_count += 1
                traffic_matrix[input_node][output_node] = math.inf
            return 1
        else:
            #print("block")
            return 1
    def traffic_matrix_free(self, traffic_matrix):
        matrix = pd.DataFrame.from_dict(traffic_matrix).to_numpy()
        matrix[matrix == math.inf] = 0
        #print(matrix)
        #print(traffic_matrix)
        if np.any(matrix):
            return True
        else:
            return False

class Connection(object):
    def __init__(self, input, output, signal_power):
        self._input = str(input)
        self._output = str(output)
        self._signal_power = float(signal_power)
        self._latency = 0.0
        self._snr = 0.0
        self._bit_rate = 0.0
    @property
    def input(self):
        return self._input
    @property
    def output(self):
        return self._output
    @property
    def signal_power(self):
        return self._signal_power
    @property
    def latency(self):
        return self._latency
    @latency.setter
    def latency(self, latency):
        self._latency = latency
    @property
    def snr(self):
        return self._snr
    @snr.setter
    def snr(self, snr):
        self._snr = snr

    @property
    def bit_rate(self):
        return self._bit_rate

    @bit_rate.setter
    def bit_rate(self, bit_rate):
        self._bit_rate = bit_rate
