# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def remain_graph(graph):
    graph_name = list(graph.keys())[0]
    graph_label = graph[graph_name][graph_name + '_label']
    G = nx.Graph()
    nodes = graph[graph_name][graph_name + '_node_label']
    for node in nodes:
        G.add_node(node, type=nodes[node], sub=1)
    edges = graph[graph_name][graph_name + '_adj']
    G.add_edges_from(edges)

    remain_G = G.copy()

    subgraphs = graph[graph_name][graph_name + '_subgraphs']
    for subgraph in subgraphs:
        vote = subgraphs[subgraph][subgraph + '_vote']
        if vote == graph_label:
            subG = nx.Graph()
            sub_edges = subgraphs[subgraph][subgraph + '_adj']
            subG.add_edges_from(sub_edges)
            for v in subG:
                remain_G.nodes[v]['sub'] += 1
    for v in G:
        if remain_G.nodes[v]['sub'] == 1:
            # remain_G.remove_node(v)
            remain_G.nodes[v]['type'] = -1
    plt.figure(figsize=(5, 3))
    plt.title(graph_name + ' label:' + str(graph_label))
    remain_pos = nx.spring_layout(remain_G, seed=1)
    remain_node_color = [color_dict[nx.get_node_attributes(remain_G, 'type')[v]] for v in remain_G]
    remain_node_size = [nx.get_node_attributes(remain_G, 'sub')[v] * 70 for v in remain_G]
    # plt.subplot(211)
    # nx.draw_networkx(G, pos, node_size=1000, node_color=node_color, edge_color='.4', width=4)
    # plt.axis('off')
    # plt.subplot(212)
    nx.draw_networkx(remain_G, remain_pos, node_size=remain_node_size, node_color=remain_node_color, with_labels=False,
                     edge_color='.4', width=3)
    plt.axis('off')
    # plt.tight_layout()
    plt.show()
    G.clear()
    return


class show_dataset_graph():
    def __init__(self, dataset="MUTAG", type="txt", npyName=''):
        if type == "txt":
            self.dataset = dataset
            self.labels = self.read_labels_from_txt()
            self.edges = self.read_edges_from_txt()
            self.nodes = self.read_nodes_from_txt()
            self.graph_labels = self.read_graph_labels_from_txt()
        elif type == "npy":
            pass

    def read_nodes_from_txt(self, fileName='graph_indicator.txt'):
        """
        从数据集原始文件中读取nodes(每个图的结点)list=>[图:[]]
        :param fileName:
        :return: nodes => {graph_indicator:[node_indicators]}
        """
        print('flag')
        full_file_name = "_".join([self.dataset, fileName])
        with open(full_file_name, 'r') as f:
            node_list = f.readlines()  # 获取 [1, 1, 2, 3,]
        nodes = dict()  # 最后需要的nodes
        for key, value in enumerate(node_list):
            key = key + 1  # 这个很关键，没有第零张图
            try:
                nodes[int(value)].append(key)
            except KeyError:
                nodes[int(value)] = [key]
        return nodes

    def graph_to_nodes(self):
        """
        获取 [{graph0_node0:label, node1:label, node2:label},{graph1_node0}, {graph1_node0:label, graph1_node2:label}]
        """
        label = self.labels
        graph_node = self.nodes
        graph_nodes_label = list() # 存放字典
        for graphIndicator in range(1, len(graph_node.keys())+1): 
            graph_nodes_label.append({x:label[x] for x in graph_node[graphIndicator]}) # 获取每个图节点的label值并存成字典append到列表里
        return graph_nodes_label


    def read_edges_from_txt(self, filName='A.txt'):
        """
        从数据集原始文件中读取邻接矩阵
        :param filName:
        :return: edges => {graph_indicator : [(2, 3), (4, 5), ...]}
        """
        ########
        graph2node = self.read_nodes_from_txt()  # 这里首先获取graph => node
        node2graph = dict()
        for key, value in graph2node.items():
            for valueInvalue in value:
                node2graph[valueInvalue] = key
        ########
        full_file_name = "_".join([self.dataset, filName])
        with open(full_file_name, 'r') as f:
            original_edges = f.readlines()  # 这里获取的是[["1, 2"], ["2, 3"]]的形式
        split_edge = lambda x: [int(y) for y in x.split(", ")]  # 将原始数据格式拆分成两个int
        original_edges = list(map(split_edge, original_edges))  # 获取的是一个列表的形式
        ########
        # 拆成子列表
        edges = dict()
        for value in original_edges:
            (node1, node2) = value
            # print(f"{node1}, {node2}")
            graph_1 = node2graph[node1]
            graph_2 = node2graph[node2]
            assert graph_1 == graph_2, f"{graph_1}, {graph_2}"  # 如果不是属于同一个图要raise error

            try:
                edges[graph_1].append(value)
            except KeyError:
                edges[graph_1] = [value]
        return edges

    def read_labels_from_txt(self, filename='node_labels.txt'):
        """
        从数据集原始文件中读取labels
        :param filename:
        :return: labels => {node_indicator: label}
        """
        full_file_name = self.dataset + '_' + filename
        with open(full_file_name, 'r') as f:
            labels_list = f.readlines()
        labels = dict()
        for key, value in enumerate(labels_list):
            key = key + 1  # 这个很关键，node从1开始
            labels[key] = int(value)
        return labels

    def dict_to_list(self, label):
        """
        将字典转换为列表，相当于用列表下标作为key值
        """
        a = [0 for _ in range(len(label.keys())+1)]
        for i in label.keys():
            a[i] = label[i]
        return a

    def read_graph_labels_from_txt(self, filename='graph_labels.txt'):
        """
        :param filename: 默认为graph_label.txt
        :return labels: {graph_indicator: label}
        """
        full_file_name = "_".join([self.dataset, filename])
        with open(full_file_name, 'r') as f:
            labels_list = f.readlines()
        labels = [-1] 
        for value in labels_list:
            labels.append(int(value))
        return labels


    def print_A_graph(self, nodes, edges, labels):
        """
        根据一张图的nodes,edges和所有图共享的labels字典绘图
        :param nodes: 标识点
        :param edges: 标识邻接矩阵
        :param label: 标识每个点的label
        :return:
        """
        G = nx.Graph()
        for node in nodes:
            G.add_node(node, type=labels[node], sub=1)
        G.add_edges_from(edges)
        remain_G = G.copy()

        nx.draw_networkx(remain_G)
        plt.axis('off')
        plt.show()
        G.clear()

    def main(self, graph_indictor=None):
        """
        输入是 一个数字时 表示输出第几张图的图
        输入是 一个元祖时 表示输出从第几张到第几张的图，包括最后一张
        不输入时 默认输出所有图
        :param graph_indictor:
        :return:
        """
        labels = self.labels
        nodes = self.nodes
        edges = self.edges
        if isinstance(graph_indictor, int):
            self.print_A_graph(nodes[graph_indictor], edges[graph_indictor], labels)
        if isinstance(graph_indictor, tuple):
            start, end = graph_indictor
            for i in range(start, end + 1):
                self.print_A_graph(nodes[i], edges[i], labels)
        elif isinstance(graph_indictor, list):
            for i in graph_indictor:
                self.print_A_graph(nodes[i], edges[i], labels)
        elif graph_indictor == None:
            graph_indictor = nodes.keys()
            for i in graph_indictor:
                self.print_A_graph(nodes[i], edges[i], labels)

if __name__ == "__main__":
    # show_graph()
    # a = show_dataset_graph("MUTAG")
    # a.main((1, 10))
    a = show_dataset_graph('DD')
    np.save('adjs_onebyone.npy', a.read_nodes_from_txt())
    # np.save('graph_node_label.npy', a.dict_to_list(a.read_labels_from_txt()))
    np.save('graph_node_labels.npy', a.graph_to_nodes())
    np.save('graphs_label.npy', a.graph_labels)
