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
from scipy.special import erfcinv as erfcinv

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
            lightpath = node.propagate(lightpath,self.label[0])
        else:
            lightpath = node.propagate(lightpath,None)
        return lightpath
    def probe(self, signal_information):
        latency = self.latency_generation()
        signal_information.add_latency(latency)
        signal_power = signal_information.signal_power
        noise = self.noise_generation(signal_power)
        signal_information.add_noise(noise)
        node = self.successive[signal_information.path[0]]
        lightpath = node.propagate(signal_information)
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
        node_json = json.load(open(json_path, 'r'))
        self._nodes = {}
        self._lines = {}
        self._weighted_path = pd.DataFrame()
        self._switching_matrix = {}
        columns_name = ["path", "channels"]
        self._route_space = pd.DataFrame(columns=columns_name)

        for node_label in node_json:
            # Create the node instance
            node_dict = node_json[node_label]
            node_dict['label'] = node_label
            node = Node(node_dict)
            if 'transceiver' in node_json[node_label].keys():
                node.transceiver = node_json[node_label]['transceiver']
            else:
                node.transceiver = 'fixed-rate'
            self._nodes[node_label] = node
            # Create the line instances
            for connected_node_label in node_dict['connected_nodes']:
                line_dict = {}
                line_label = node_label + connected_node_label
                line_dict['label'] = line_label
                node_position = np.array(node_json[node_label]['position'])
                connected_node_position = np.array(node_json[connected_node_label]['position'])
                line_dict['length'] = np.sqrt(
                    np.sum((node_position - connected_node_position) ** 2)
                )
                line = Line(line_dict)
                self._lines[line_label] = line

            self._switching_matrix[node_label] = node_dict['switching_matrix']

    @property
    def nodes(self):
        return self._nodes

    @property
    def lines(self):
        return self._lines

    @property
    def weighted_path(self):
        return self._weighted_path

    @weighted_path.setter
    def weighted_path(self, weighted_path):
        self._weighted_path = weighted_path

    @property
    def route_space(self):
        return self._route_space

    @route_space.setter
    def route_space(self, route_space):
        self._route_space = route_space

    @property
    def switching_matrix(self):
        return self._switching_matrix

    @switching_matrix.setter
    def switching_matrix(self, switching_matrix):
        self._switching_matrix = switching_matrix

    def draw(self):
        nodes = self.nodes
        for node_label in nodes:
            n0 = nodes[node_label]
            x0 = n0.position[0]
            y0 = n0.position[1]
            plt.plot(x0, y0, 'go', markersize=10)
            plt.text(x0 + 20, y0 + 20, node_label)
            for connected_node_label in n0.connected_nodes:
                n1 = nodes[connected_node_label]
                x1 = n1.position[0]
                y1 = n1.position[1]
                plt.plot([x0, x1], [y0, y1], 'b')
        plt.title('Network')
        plt.show()

    def find_paths(self, label1, label2):
        cross_nodes = [key for key in self.nodes.keys()
                       if ((key != label1) & (key != label2))]
        cross_lines = self.lines.keys()
        inner_paths = {'0': label1}
        for i in range(len(cross_nodes) + 1):
            inner_paths[str(i + 1)] = []
            for inner_path in inner_paths[str(i)]:
                inner_paths[str(i + 1)] += [inner_path + cross_node for cross_node in cross_nodes
                                            if ((inner_path[-1] + cross_node in cross_lines) & (
                            cross_node not in inner_path))]
        paths = []
        for i in range(len(cross_nodes) + 1):
            for path in inner_paths[str(i)]:
                if path[-1] + label2 in cross_lines:
                    paths.append(path + label2)
        return paths

    def connect(self):
        nodes_dict = self.nodes
        lines_dict = self.lines

        for node_label in nodes_dict:
            node = nodes_dict[node_label]
            '''Assinging switching matrix by using deepcopy to avoid 
            reflection of modification in the original switching matrix of Network class '''
            node.switching_matrix = copy.deepcopy(self.switching_matrix[node_label])

            for connected_node in node.connected_nodes:
                line_label = node_label + connected_node
                line = lines_dict[line_label]
                line.successive[connected_node] = nodes_dict[connected_node]
                node.successive[line_label] = lines_dict[line_label]

    def propagate(self, lightpath):
        start_node = self.nodes[lightpath.path[0]]
        propagated_signal_information = start_node.propagate(lightpath, None)
        return propagated_signal_information

    def find_best_snr(self, node_input, node_output):
        if node_input != node_output:
            my_df = self.weighted_path
            my_df.sort_values(by=['snr'], inplace=True, ascending=False)
            my_df_filtered = my_df[(my_df['path'].str[0] == node_input) & (my_df['path'].str[-1] == node_output)]
            for i in my_df_filtered.values:
                path = i[0]  # path
                channel = self.channel_free(path)
                if channel is not None:
                    # [best_path, latency, noise, snr], best_path, channel selected for the best path
                    return i, i[0], channel
        return None, None, None

    ''' Method to find the index of the first channel free for the specified path'''
    def channel_free(self, path):
        path_in_route_space = self.route_space[self.route_space['path'] == path]
        for i in range(param.channels):
            if path_in_route_space['channels'].values[0][i] == 1:
                return i
        return None

    def find_best_latency(self, node_input, node_output):
        if node_input != node_output:
            my_df = self.weighted_path
            my_df.sort_values(by=['latency'], inplace=True, ascending=True)
            my_df_filtered = my_df[(my_df['path'].str[0] == node_input) & (my_df['path'].str[-1] == node_output)]
            for i in my_df_filtered.values:
                path = i[0]
                channel = self.channel_free(path)
                if channel is not None:
                    # [best_path, latency, noise, snr], best_path, channel selected for the best path
                    return i, i[0], channel
        return None, None, None

    def stream(self, connections, label='latency'):
        for connection in connections:
            if label == 'snr':
                best_path_array, best_path, channel = self.find_best_snr(connection.input, connection.output)
            else:
                best_path_array, best_path, channel = self.find_best_latency(connection.input, connection.output)

            if best_path is not None:
                # path string without "->"
                path_label = ''
                for index in range(0, len(best_path), 3):
                    path_label += best_path[index]

                lightpath = Lightpath(connection.signal_power, path_label, channel)
                bit_rate = self.calculate_bit_rate(lightpath, self.nodes[path_label[0]].transceiver)
                if bit_rate == 0:
                    connection.snr = 0
                    connection.latency = -1  # None
                    connection.bit_rate = 0
                else:
                    self.propagate(lightpath)
                    connection.snr = self.snr_dB(lightpath)
                    connection.latency = lightpath.latency
                    connection.bit_rate = bit_rate
                    # Updating routing space after lightpath propagation
                    self.update_routing_space(best_path)  # 0= route space not empty
            else:
                ''' if there is no best path for snr and latency'''
                connection.snr = 0
                connection.latency = -1  # None

    def snr_dB(self, lightpath):
        return (10 * np.log10(lightpath.signal_power / lightpath.noise_power))

    def update_routing_space(self, best_path):
        df = self.weighted_path['path'].str.replace('->','')
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
            idx = self.weighted_path.loc[df == path].index.values[0]
            self.route_space.iloc[idx] = occupancy
        return None

    def restore_network(self):
        self.route_space = self.route_space[0:0]
        nodes_dict = self.nodes
        lines_dict = self.lines
        for node_label in nodes_dict:
            node = nodes_dict[node_label]
            node.switching_matrix = copy.deepcopy(self.switching_matrix[node_label])

        for line_label in lines_dict:
            line = lines_dict[line_label]
            # print(line.state)
            line.state = np.ones(param.channels, np.int8)  # channel free

        self.update_routing_space(None)

    def calculate_bit_rate(self, lightpath, strategy):
        path_label = ''
        for node in lightpath.path:
            path_label += node + '->'
        gsnr_dB = self.weighted_path[self.weighted_path['path'] == path_label[:-2]]['snr'].values[0]  # dB
        gsnr = 10 ** (gsnr_dB / 10)
        #print("GSNR linear: ", gsnr, " GSNR dB: ", gsnr_dB)
        if strategy == 'fixed_rate':
            if gsnr >= 2 * ((erfcinv(2 * param.BERt)) ** 2) * lightpath.Rs / param.Bn:
                return 100e9
            else:
                return 0
        elif strategy == 'flex_rate':
            if gsnr < 2 * ((erfcinv(2 * param.BERt)) ** 2) * lightpath.Rs / param.Bn:
                return 0
            elif gsnr < 14 / 3 * ((erfcinv(3 / 2 * param.BERt)) ** 2) * lightpath.Rs / param.Bn:
                return 100e9
            elif gsnr < 10 * ((erfcinv(8 / 3 * param.BERt)) ** 2) * lightpath.Rs / param.Bn:
                return 200e9
            else:
                return 400e9
        elif strategy == 'shannon':
            return 2 * lightpath.Rs * np.log2(1 + (gsnr * param.Bn / lightpath.Rs))

    def request_generation_traffic_matrix(self, traffic_matrix, connections, signal_power):
        nodes = list(self.nodes.keys())
        while True:
            input_rand = rand.choice(nodes)
            output_rand = rand.choice(nodes)
            ''' Generating connections only for input_node != output_node and
             if the element corresponding to input, output node in the traffic matrix has still available traffic to allocate'''
            if input_rand != output_rand and traffic_matrix[input_rand][output_rand] != 0 and \
                    traffic_matrix[input_rand][output_rand] != math.inf:
                break
        connection = Connection(input_rand, output_rand, signal_power)
        current_connections = [connection]
        self.stream(current_connections, 'snr')
        connections.append(connection)

        # Updating traffic matrix if a best path is found in the stream() method
        if connection.snr != 0:
            # Bit rate request for the connection is completely satisfied
            if connection.bit_rate >= traffic_matrix[input_rand][output_rand]:
                traffic_matrix[input_rand][output_rand] = 0
                return 1  # decrement, capacity guaranteed
            else:
                # Updating the remaining capability after having satisfied the capability for the current connection
                traffic_matrix[input_rand][output_rand] -= connection.bit_rate
                return 0
        else:
            traffic_matrix[input_rand][output_rand] = math.inf
        # If is not possible to define a suitable path for the connection,
        # decrement the number of all possible connections by returning 1
        return 1  # decrement
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
