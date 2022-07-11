import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import c
from core import parameters as param

class SignalInformation(object):
    def __init__(self, power, path):
        self._signal_power = float(power)
        self._noise_power = float(0.0)
        self._latency = float(0.0)
        self._path = path
    @property
    def signal_power(self):
        return self._signal_power
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
    @property
    def channel(self):
        return self._channel


class Node(object):
    def __init__(self, node_dict):
        self._label = node_dict['label']
        self._position = node_dict['position']
        self._connected_nodes = node_dict['connected_nodes']
        self._successive = {}
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
    def propagate(self, lightpath):
        path = lightpath.path
        if len(path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]
            lightpath.next()
            lightpath = line.propagate(lightpath)
        return lightpath

class Line(object):
    def __init__(self, line_dict):
        self._label = line_dict['label']
        self._length = line_dict['length']
        self._successive = {}
        self._state = ['free'] * param.channels
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
    def latency_generation(self):
        latency = self.length / (c * 2 / 3)
        return latency
    def noise_generation(self, signal_power):
        noise = 1e-3 * signal_power * self.length
        return noise
    def propagate(self, lightpath):
        latency = self.latency_generation()
        lightpath.add_latency(latency)
        signal_power = lightpath.signal_power
        noise = self.noise_generation(signal_power)
        lightpath.add_noise(noise)
        node = self.successive[lightpath.path[0]]
        lightpath = node.propagate(lightpath)
        if type(lightpath) == Lightpath:
            self.state[lightpath.channel] = 'occupied'
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

class Network(object):
    def __init__(self, json_path):
        node_json = json.load(open(json_path,'r'))
        self._nodes = {}
        self._lines = {}
        self._weighted_paths = pd.DataFrame()
        self._route_space = pd.DataFrame()
        for node_label in node_json:
            # create node instance
            node_dict = node_json[node_label]
            node_dict['label'] = node_label
            node = Node(node_dict)
            self._nodes[node_label] = node
            #create line instances
            for connected_node_label in node_dict['connected_nodes']:
                line_dict = {}
                line_label = node_label + connected_node_label
                line_dict['label'] = line_label
                node_position = np.array(node_json[node_label]['position'])
                connected_node_position = np.array(node_json[connected_node_label]['position'])
                line_dict['length'] = np.sqrt(np.sum((node_position-connected_node_position))**2)
                line = Line(line_dict)
                self._lines[line_label] = line
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

                signal_information = SignalInformation(1, path)
                signal_information = self.probe(signal_information)
                latencies.append(signal_information.latency)
                noises.append(signal_information.noise_power)
                snrs.append(10 * np.log10(signal_information.signal_power / signal_information.noise_power))
                path_state.append('free')
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
    def draw(self):
        nodes = self.nodes
        for node_label in nodes:
            n0 = nodes[node_label]
            x0 = n0.position[0]
            y0 = n0.position[1]
            plt.plot(x0,y0,'go',markersize=10)
            plt.text(x0+20,y0+20,node_label)
            for connected_node_label in n0.connected_nodes:
                n1 = nodes[connected_node_label]
                x1 = n1.position[0]
                y1 = n1.position[1]
                plt.plot([x0,x1], [y0,y1], 'b')
        plt.title('Network')
        plt.show()
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
            for connected_node in node.connected_nodes:
                line_label = node_label + connected_node
                line = lines_dict[line_label]
                line.successive[connected_node] = nodes_dict[connected_node]
                node.successive[line_label] = lines_dict[line_label]
    def propagate(self, lightpath):
        path = lightpath.path
        start_node = self.nodes[path[0]]
        propagated_signal_information = start_node.propagate(lightpath)
        return propagated_signal_information
    def probe(self, signal_information):
        path = signal_information.path
        start_node = self.nodes[path[0]]
        propagated_signal_information = start_node.propagate(signal_information)
        return propagated_signal_information
    def find_best_snr(self, input_label, output_label):
        paths_df = self.weighted_paths
        paths_df = paths_df.loc[(paths_df['path'].str[0] == input_label) & (paths_df['path'].str[-1] == output_label)]
        paths_df = paths_df.sort_values(['snr'], inplace=False, ascending=False)
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
                self.propagate(lightpath)
                self.update_route_space(best_path, channel)
                #print("connect:",connection.input,"->",connection.output, "of path:",best_path,"using channel:", channel)
                if label == 'snr':
                    connection.snr = 10 * np.log10(lightpath.signal_power / lightpath.noise_power)
                else:
                    connection.latency = lightpath.latency
            else:
                df = self.route_space
                df = df.loc[(self.weighted_paths['path'] == best_path.replace('', '->')[2:-2])]
                #print("Not possible to connect:",connection.input,"->",connection.output)
                #print("Occupation of", best_path)
                #print(df)
                connection.snr = 0.0
                connection.latency = 0.0



    def find_channel(self, path):
        channel = None
        for ch in range(0, param.channels):
            df = self.route_space
            df = df.loc[(self.weighted_paths['path'] == path.replace('','->')[2:-2])]
            if df['Ch.'+str(ch)].values == 'free':
                channel = ch
                break
        return channel
    def update_route_space(self, best_path, channel):
        df = self.route_space
        for i in range(0,len(best_path)-1):
            df_idx = self.weighted_paths['path'].loc[self.weighted_paths['path'].str.replace('->','').str.contains(best_path[i:i+2])].index
            df['Ch.'+str(channel)].iloc[df_idx] = 'occupied'
            #print(df['Ch.'+str(channel)].iloc[df_idx])
        #print("occupying Ch.",channel,"of path:",best_path)
        return None

class Connection(object):
    def __init__(self, input, output, signal_power):
        self._input = str(input)
        self._output = str(output)
        self._signal_power = float(signal_power)
        self._latency = 0.0
        self._snr = 0.0
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
    @property
    def snr(self):
        return self._snr
    @latency.setter
    def latency(self, latency):
        self._latency = latency
    @snr.setter
    def snr(self, snr):
        self._snr = snr
