import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import random
from scipy.special import comb

random.seed(0)

class ModelData:
    def __init__(self, dataset):
        self.init_graph = pd.read_csv(dataset)
        self.nodes = self.init_graph[['source', 'target']]


def graphA(links_data):
    # creates undirected unweighted graph from edges (source,terget)
    graph = nx.from_pandas_edgelist(links_data.nodes, 'source', 'target')
    return graph


def calc_loglog_data(G):
    y = nx.degree_histogram(G)  # A list of frequencies of degrees. The degree values are the index in the list.
    # y=[0, 2495, 1208, 637, 362, 233, 152, 129, 85, 93, 64, 39, 44, 32, 35, 19, 17, 21, 7, 17, 18, 16, 12, 11, 9, 5...
    # for example there are 2495 nodes with degree of 1

    x = []  # the vector of degrees that we have in graph G
    for i in range(len(y)):
        if y[i] != 0:
            x = x + [i]
    # x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,..

    y[:] = (value for value in y if
            value != 0)  # replace the content of y (y[:]- all the content) with the content of y without 0

    # np.log(x) creates log version of x
    # np.log(y) creates log version of y
    return np.log(x), np.log(y)


def calc_best_power_law(G):
    x_data, y_data = calc_loglog_data(G)

    line = np.polyfit(x_data, y_data, 1)  # creates the linear regression line

    alpha = -(line[0])  # The slope
    beta = np.exp(line[1])  # Point of intersection with y-axis

    # alpha=1.7945971770334244, beta=2966.639755862869
    return alpha, beta


def plot_histogram(G, alpha, beta):
    # calculates log-log graph
    x_data, y_data = calc_loglog_data(G)
    plt.scatter(x_data, y_data)
    plt.xlabel('Log Degree')
    plt.ylabel('Log Number of nodes')

    # calculates linear line
    x = np.linspace(0, 4.5)
    y = np.log(beta) - x * alpha  # like the descreption in the book (18.2) log y=log(beta)-log(alpha)*x
    plt.plot(x, y, 'r')

    plt.title('G-nodes Histogram in Log-Log Graph')
    plt.savefig('power_law_distribution.png')
    return


def G_features(G):  #########return dict of dicts ########
    # calculates Closeness Centrality and Betweenness Centrality
    G_betweeness = nx.betweenness_centrality(G)
    G_closeness = nx.closeness_centrality(G,wf_improved=False)
    return {'Closeness': G_closeness, 'Betweeness': G_betweeness}


def create_undirected_weighted_graph(links_data, users_data, question):  #########return Graph ########
    # creates undirected weighted graph

    WG = nx.from_pandas_edgelist(links_data.init_graph, 'source', 'target', ['weight'])
    # the key to each node is column question
    infected_dict = users_data.set_index('node')[question].to_dict()
    # Each node answers the question of whether it is infected
    nx.set_node_attributes(WG, infected_dict, 'infected?')
    #nx.draw_networkx(WG)
    return WG


def run_k_iterations(WG, k, Threshold):
    tmp = nx.get_node_attributes(WG, 'infected?')
    tmp_nodes = list(WG.nodes)
    # creates the initial list of infected nodes
    s0_list = []
    for key, value in tmp.items():
        if value == 'YES' or value == 'Yes':
            s0_list.append(key)
    S = {}
    S[0] = s0_list
    k_iteration = 1
    exposure = 0

    while k_iteration <= k:
        S_list = []
        i = 0
        while i < len(WG.nodes):
            # checks for non infected nodes
            if WG.nodes[tmp_nodes[i]]['infected?'] == 'YES':
                i += 1
                continue
            for neighbor in list(WG.neighbors(tmp_nodes[i])):
                #checks if node i neighbors' are infected
                if WG.nodes[neighbor]['infected?'] == 'YES':
                    # The amount of weights of the nodes that have been infected so far
                    exposure += WG.edges[tmp_nodes[i], neighbor]['weight']
                if exposure >= Threshold: #if infected
                    S_list.append(tmp_nodes[i])
                    break

            i += 1
            exposure = 0
        for j in S_list:
            WG.nodes[j]['infected?'] = 'YES'

        S[k_iteration] = S_list
        k_iteration += 1

    return S


# return a dictionary of with the branching factors R1,...Rk
def calculate_branching_factor(S, k):

    BF={}
    j=1
    while j<=k:
        BF[j]=len(S[j])/len(S[j-1])
        j+=1
    return BF


# return a dictionary of the h nodes with the highest degree, sorted by decreasing degree
# {index : node}
def find_maximal_h_deg(WG, h):
    nodes_dict = {}
    deg = dict(WG.degree())
    for j in range(h):
        tmp_max = max(deg, key=deg.get)
        del deg[tmp_max]
        nodes_dict[tmp_max] = WG.degree[tmp_max]

    return nodes_dict


# return a dictionary of all nodes with their clustering coefficient
# {node : CC}
def calculate_clustering_coefficient(WG, nodes_dict):

    nodes_dict_clustering = {}
    for i in nodes_dict.keys():
        # List of neighbors of the node i
        tmp_nodes=[n for n in nx.neighbors(WG, i)]
        tmp_nodes_len=len(tmp_nodes)
        count = 0
        #Searches between the neighbors of the node=nodes_list[i], nodes that are themselves neighbors
        for n1 in tmp_nodes:
            for n2 in tmp_nodes:
                if WG.has_edge(n1,n2):
                    count+=1

        nodes_dict_clustering[i] = (count * 2) / (tmp_nodes_len * (tmp_nodes_len - 1))

    return nodes_dict_clustering


def infected_nodes_after_k_iterations(WG, k, Threshold):
    S=run_k_iterations(WG,k,Threshold)
    count=0
    for i in range(len(S)):
        count+=len(S[i])

    #print('count=',count)
    return count


# return the first [number] nodes in the list
def slice_dict(dict_nodes, number):
    nodes_list=list(dict_nodes)
    slice=nodes_list[:number]

    return slice


# remove all nodes in [nodes_list] from the graph WG, with their edges, and return the new graph
def nodes_removal(WG, nodes_list):
    for i in range(len(nodes_list)):
        WG.remove_node(nodes_list[i])
    #print('neighbors', neighbors)
    return WG


# plot the graph according to Q4 , add the histogram to the pdf and run the program without it
def graphB(number_nodes_1, number_nodes__2, number_nodes_3):
    # implement for Q4

    x1,y1=zip(* (sorted(number_nodes_1.items())))
    x2,y2=zip(*(sorted(number_nodes__2.items())))
    x3,y3=zip(*(sorted(number_nodes_3.items())))


    plt.title('graphB')
    plt.plot(x1,y1,label='RANDOM')
    plt.plot(x2,y2,label='ASC')
    plt.plot(x3,y3,label='DSC')
    plt.ylabel('final number of infected nodes')
    plt.xlabel('number of removed nodes')
    plt.legend()
    plt.savefig('graphB.png')
    return

