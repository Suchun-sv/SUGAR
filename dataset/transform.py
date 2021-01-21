from keras.utils import to_categorical
import random
import os
import numpy as np
import re
from sklearn import preprocessing
import configparser
import argparse
min_max_scaler = preprocessing.MinMaxScaler()
######################################################################
# 命令行参数初始化
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="MUTAG", dest='dataset')
args = parser.parse_known_args()[0]
######################################################################
# 读取配置参数
os.chdir('./{}'.format(args.dataset))
print('dataset:', args.dataset)
config = configparser.ConfigParser()  
config.read('.config')
getConfig =lambda x : config.get('config', x)
ds = getConfig('ds')
SUB_size = int(getConfig('SUB_size'))
min = int(getConfig('min'))
# max_graph_nodes = int(getConfig('max_graph_nodes'))
max_graph_nodes = 100
cate = int(getConfig('cate'))
class_size = int(getConfig('class_size'))
######################################################################

class graphParser(object):
    def __init__(self):
        self.ds = ds
        self.SUB_size = SUB_size
        self.min = min
        self.cate = cate
        self.class_size = class_size
        self.max_graph_nodes = max_graph_nodes

        #############################################################
        # 从原始数据集中获取多种参数
        with open(self.ds+'_A.txt', 'r') as f:
            a = f.readlines()
            self.m = len(a)

        with open(self.ds+'_graph_labels.txt', 'r') as f:
            a = f.readlines()
            self.N = len(a)

        with open(self.ds+'_node_labels.txt', 'r') as f:
            a = f.readlines()
            self.nodeIndex2label = {}
            for index, item in enumerate(a):
                self.nodeIndex2label[index] = int(item)
            self.n = len(a)
        #############################################################
        # 输出获取的多个参数
        print("m:{},N:{},n:{}".format(self.m, self.N, self.n))

    def ex_which_graph(self):  # graph's nodes
        """
        这个就是确定是哪个节点是哪个图的
        """
        d_n2g = {}  # start from 0
        with open(self.ds+'_graph_indicator.txt') as f:  # 打开graph_indicator
            index = 0  # index 标记的是行数
            line = f.readline()
            while line:
                d_n2g[index] = int(line)
                index += 1
                line = f.readline()
        # print(d_n2g)
        d_gns = {k: [] for k in range(self.N)}
        for it in d_n2g:
            d_gns[d_n2g[it]-1].append(it)  # start from 0
        # d_gns: 就是每个图=>list(节点)
        return d_gns

    def ex_edges(self):
        d_es = {k: [] for k in range(self.n)}
        with open(self.ds+'_A.txt') as f:
            # 首先_A这个文件存放的就是任意多条边
            line = f.readline()
            while line:
                # print(line)
                line = line.replace(' ', '')
                s = re.search('[0-9]{1,}', line).group()
                d = re.search('[,][0-9]{1,}', line).group()
                d = d.strip(',')
                # 也就是把源点=>list(目的点),这里考虑的可能还是有向图
                d_es[int(s)-1].append(int(d)-1)
                line = f.readline()
        adj = np.zeros((self.n, self.n))
        for it in d_es:
            for its in d_es[it]:
                adj[it][its] = 1
        # print(d_es)
        return d_es, adj

    def ex_labels(self, d_gns):
        d_gl = {}
        glabs = np.zeros((self.N, self.class_size))
        with open(self.ds+'_graph_labels.txt') as f:
            # 首先这个文件就是用来分类的
            #######################################################################
            # 鉴于每个数据集的数据类型命名方式都有着差异，考虑重新给类别命名
            old_cate_name2new_cate_new = {} # 定义空字典
            new_cate_name = 0 # 新类型名称从0开始
            #######################################################################
            line = f.readline()
            index = 0
            while line:
                d_gl[index] = int(line)
                if d_gl[index] not in old_cate_name2new_cate_new.keys():
                    old_cate_name2new_cate_new[d_gl[index]] = new_cate_name
                    new_cate_name+=1
                glabs[index] = to_categorical(old_cate_name2new_cate_new[d_gl[index]], self.class_size) # 对新类别进行one_hot编码
                index += 1
                line = f.readline()
            #glabs[0] = [0,1]
            #glabs[1] = [1,0]
        return glabs

    def gen_adjmatrix(self, d_gns, adj_com):
        adjs = []
        index = 0
        for it in d_gns:
            gsize = len(d_gns[it])
            adj = adj_com[index:index+gsize, index:index+gsize]
            adjs.append(adj)
            index += gsize
        return adjs

    def gen_adj_onebyone(self, d_gns, adj_com):
        adjs = []
        index = 0
        for it in d_gns:
            gsize = len(d_gns[it])
            adj = []
            for nodeA in range(index, index+gsize):
                for nodeB in range(index+1, index+gsize):
                    if adj_com[nodeA, nodeB] == 1:
                        adj.append(tuple(sorted([nodeA, nodeB]))) # 按一定顺序插入
            index += gsize
            adjs.append(list(set(adj))) # 去重
        return adjs

    def nodeIndex2nodelabel(self, d_gns):
        nodeLabel = []

        for graph in d_gns.values():  # graph 应该是一个list(nodeIndex)
            # print(graph)
            # print('self.nodeIndex2label', self.nodeIndex2label)
            tempList = {x:self.nodeIndex2label[x] for x in graph}
            nodeLabel.append(tempList)
        return nodeLabel

    def main(self, save=True):
        d_gns = {}  # graph's nodes
        d_es = {}  # edges
        d_gl = {}  # graph-label
        glabs = []
        adj_com = []

        d_gns = self.ex_which_graph()  # 存的是每个图=>list(节点)
        d_es, adj_com = self.ex_edges()  # d_es 源点=>list(目标点)
        glabs = self.ex_labels(d_gns)  # 存的是每个图的分类=> [0,1] [1,0]
        # print('d_gns', d_gns)
        nodeLabel = self.nodeIndex2nodelabel(d_gns)
        adjs = []
        adjs = self.gen_adjmatrix(d_gns, adj_com)  # 存的是每个图的邻接矩阵
        adjs_onebyone = self.gen_adj_onebyone(d_gns, adj_com)

        self.d_gns = d_gns
        if save:
            f = open('edges.txt', 'w')
            f.write(str(d_es))
            f.close()

            f = open('nodes_in_graphs.txt', 'w')
            f.write(str(d_gns))
            f.close()

            np.save('graphs_label.npy', glabs)
            np.save('d_gns.npy', list(d_gns.values()))
            # print(glabs.shape)
            np.save('{}adjs.npy'.format(self.N), adjs)
            if not os.path.exists('graph'):
                os.mkdir('graph')
            np.save('graph/graphs_label.npy', glabs)
            np.save('graph/graph_node_labels.npy', nodeLabel)
            np.save('graph/adjs_onebyone.npy', adjs_onebyone)

        return adjs, d_es


class secondParse(graphParser):
    def __init__(self, adjs=None, dict_edges=None):
        super(secondParse, self).__init__()
        if not adjs:
            self.adjs = np.load('{}adjs.npy'.format(N), allow_pickle=True)  # 每个图的邻接矩阵
        else:
            self.adjs = adjs

        if not dict_edges:
            with open('edges.txt') as f:
                self.dict_edges = eval(f.read())
        else:
            self.dict_edges = dict_edges

    def bfs(self, adj, s):  # for a graph
        n = self.n
        m = self.m
        N = self.N
        SUB_size = self.SUB_size
        min = self.min
        sub = []
        # The graph's adjacency matrix
        matrix = adj
        # The visited array
        visited = [0 for _ in range(len(adj[0]))]
        # Add the start node to the queue
        # Node 0 in this case
        queue = [s]
        # Set the visited value of node 0 to visited
        visited[s] = 1
        # Dequeue node 0
        node = queue.pop(0)
        # print(node)
        sub.append(node)
        while True:
            for x in range(0, len(visited)):
                # Check is route exists and that node isn't visited
                if matrix[node][x] == 1 and visited[x] == 0:
                    # Visit node
                    visited[x] = 1
                    # Enqueue element
                    queue.append(x)

            # When queue is empty, break
            if len(queue) == 0:
                break
            else:
                # Dequeue element from queue
                newnode = queue.pop(0)
                node = newnode
                # print(node)
                sub.append(node)
                if len(sub) == SUB_size:  # 这里提前结束，就是规定了子图最大的大小
                    sub.sort()
                    return sub
        sub.sort()
        return sub

    def gen_subs(self, adj):  # for a graph
        n = self.n
        m = self.m
        N = self.N
        SUB_size = self.SUB_size
        min = self.min
        subs = []
        # print(adj[0])
        m = {}
        for s in range(len(adj[0])):  # 这里的len(adj[0])实际上就是结点个数
            # print(bfs(adj, s))#这里就是求联通子图的结点list
            if tuple(self.bfs(adj, s)) not in m.keys():
                subs.append(self.bfs(adj, s))
                m[tuple(self.bfs(adj, s))] = 1

        subs = subs[:min]   
        # print(len(subs))# 这一行问题很大，这个应该是每个图的子图多少
        return subs

    def gen_upperlevel_graph(self, sub):
        n = self.n
        m = self.m
        N = self.N
        SUB_size = self.SUB_size
        min = self.min
        # ADJ_of_a_graph=gen_upperlevel_graph(sub_of_a_graph)
        adj = np.zeros((len(sub), len(sub)))
        for i in range(len(sub)):
            for j in range(i+1, len(sub)):
                # 这个就是两个子图之间有多少个节点重复
                same = [it for it in sub[i] if it in sub[j]]
                # print(same)
                adj[i][j] = len(same)
            # print('before', adj[i])
            adj[i] = min_max_scaler.fit_transform(
                adj[i].reshape(-1, 1)).reshape(-1)
            # print('after', adj[i])
        # print(adj)
        for i in range(len(sub)):
            adj[i][i] = 1
        return adj

    def gen_subgraphs_adj_onebyone(self, all_adj, subs, nodeList):
        n = self.n
        m = self.m
        N = self.N
        SUB_size = self.SUB_size
        min = self.min
        minNode = np.min(nodeList)
        adjs = []
        # print('subs', len(subs))
        for key, sub in enumerate(subs):
            # print('sub', sub)
            index = 0
            gsize = len(sub)
            adj = []
            for nodeA in sub:
                for nodeB in sub:
                    if all_adj[nodeA][nodeB] == 1 or all_adj[nodeB][nodeA] == 1:
                        adj.append(tuple(sorted([minNode+nodeA, minNode+nodeB]))) # 排序
            adjs.append(list(set(adj))) # 去重
        return adjs
                

    def gen_subgraphs_adj(self, all_adj, subs):
        n = self.n
        m = self.m
        N = self.N
        SUB_size = self.SUB_size
        min = self.min
        # subadjs_of_a_graph=gen_subgraphs_adj(adj,sub_of_a_graph,dict_edges)
        adjs = []
        for sub in subs:
            adj = np.zeros((SUB_size, SUB_size))
            i = 0
            for i in range(len(sub)):  # 这里就是利用了整张图的邻接矩阵来生成子图的邻接矩阵
                for j in range(len(sub)):
                    # for it in sub:
                    # for its in d[it]:
                    if all_adj[sub[i]][sub[j]] == 1:
                        adj[i][j] = 1
                # print('before', adj[i])
                adj[i] = min_max_scaler.fit_transform(
                    adj[i].reshape(-1, 1)).reshape(-1)  # 这个暂时看不出来是干什么的，前后好像都没变化
                # print('after',adj[i])
            for i in range(len(sub)):
                adj[i][i] = 1
            adjs.append(adj.copy())
        return adjs

    def main(self, save=True):
        #######################################################################
        n = self.n
        m = self.m
        N = self.N
        SUB_size = self.SUB_size
        min = self.min
        adjs = self.adjs
        self.d_gns = np.load('d_gns.npy', allow_pickle=True)
        #######################################################################
        ADJs = []
        subs_of_graphs = []
        subadjs_of_graphs = []
        subadjs_of_graphs_onebyone = []
        min_max_scaler = preprocessing.MinMaxScaler()
        #######################################################################
        for i in range(len(adjs)):
            if i % 100 == 0:
                print(i)
            adj = adjs[i]
            sub_of_a_graph = self.gen_subs(adj)
            # print(sub_of_a_graph)
            # print(len(sub_of_a_graph))
            # [[[子图1], [子图2], [子图3]], []]
            subs_of_graphs.append(sub_of_a_graph)
            subadjs_of_a_graph = self.gen_subgraphs_adj(
                adj, sub_of_a_graph)  # 这个就是根据大邻接矩阵生成小邻接矩阵
            subadjs_of_a_graph_onebyone = self.gen_subgraphs_adj_onebyone(adj, sub_of_a_graph, self.d_gns[i])
            subadjs_of_graphs_onebyone.append(subadjs_of_a_graph_onebyone)
            subadjs_of_graphs.append(subadjs_of_a_graph)
            # print(subadjs_of_a_graph)
            ADJ_of_a_graph = self.gen_upperlevel_graph(
                sub_of_a_graph)  # 这个是获取的就比较奇怪了，是求两个子图节点的重复
            # print(ADJ_of_a_graph)
            # print(ADJ_of_a_graph.shape)
            # 每次都是一个len(sub_of_a_graph),len(sub_of_a_graph)的矩阵,而len(sub)应该是小于等于SUB_size的
            ADJs.append(ADJ_of_a_graph)

            # sadjs=np.zeros((min,SUB_size,SUB_size))
            # # print(len(sub_of_a_graph))
            # for j in range(len(subadjs_of_a_graph)):
            #     if j == min:
            #         break
            #     # print(subadjs_of_a_graph.size)
            #     sadjs[j][:len(subadjs_of_a_graph[j])][:len(subadjs_of_a_graph[j])]=subadjs_of_a_graph[j].copy()# 这个就是把子图list放在一个大矩阵里

            # for j in range(min):
            #     for k in range(SUB_size):
            #         sadjs[j][k][k]=1

            # np.save('subadj/sub_adj' + str(i) + '.npy', sadjs)# 对每个图都有这么一个(sub_max_num, sub_size, sub_size)的矩阵
            # except:
            #    print('error:',i)

        # exit(0)
        # print('存储完毕')
        if save:
            np.save('subs_of_graphs.npy', subs_of_graphs)
            np.save('subadjs_of_graphs.npy', subadjs_of_graphs)
            np.save('ADJs.npy', ADJs)
            if not os.path.exists('./graph'):
                os.mkdir('./graph')
            np.save('./graph/subadjs_of_graphs_onebyone.npy', subadjs_of_graphs_onebyone)

        sadjs = np.zeros((N, min, SUB_size, SUB_size))
        sus = np.zeros((N, min, SUB_size))
        As = np.zeros((N, min, min))

        if save:
            subs_of_graphs = np.load('subs_of_graphs.npy', allow_pickle=True)
            subadjs_of_graphs = np.load(
                'subadjs_of_graphs.npy', allow_pickle=True)
            ADJs = np.load('ADJs.npy', allow_pickle=True)

        for i in range(N):

            # print(ADJs.shape)
            for j in range(len(ADJs[i])):
                As[i][j][:len(ADJs[i][j])][:len(
                    ADJs[i][j])] = ADJs[i][j].copy()
            for j in range(len(subadjs_of_graphs[i])):
                # print(sadjs.shape)
                sadjs[i][j][:len(subadjs_of_graphs[i][j])][:len(
                    subadjs_of_graphs[i][j])] = subadjs_of_graphs[i][j].copy()
            for j in range(len(subs_of_graphs[i])):
                sus[i][j][:len(subs_of_graphs[i][j])
                          ] = subs_of_graphs[i][j].copy()

        for i in range(N):
            for j in range(min):
                As[i][j][j] = 1
                for k in range(SUB_size):
                    sadjs[i][j][k][k] = 1

        if save:
            np.save('adj.npy', As)
            np.save('sub_adj.npy', sadjs)
            np.save('subindexs_of_graphs.npy', sus)
        return sus

    def third(self, sus=None):
       # 读取需要的矩阵
        if sus == None:
            sus = np.load('subindexs_of_graphs.npy', allow_pickle=True)

        with open('nodes_in_graphs.txt') as f:
            dict_gn = eval(f.read())

        labs = self.ex_labels()
        features = np.zeros((self.N, self.min, self.SUB_size, self.cate))

        for i in range(features.shape[0]):
            for j in range(features.shape[1]):
                for k in range(features.shape[2]):
                    len_x = int(sus[i][j][k])

                    features[i][j][k] = labs[i][len_x]

        # print(features)
        np.save('features.npy', features)

    def ex_labels(self):
        labs = {k: [] for k in range(self.N)}
        nodes = np.zeros(self.n)

        with open('nodes_in_graphs.txt') as f:
            dict_gn = eval(f.read())

        with open(self.ds + '_node_labels.txt') as f:
            line = f.readline()
            index = 0
            while line:
                nodes[index] = int(line)
                index += 1
                line = f.readline()

        i = 0
        for graph in dict_gn:
            labs[i] = to_categorical(
                [nodes[it]-1 for it in dict_gn[graph]], self.cate)
            i += 1
        return labs


if __name__ == "__main__":
    paser = graphParser()
    # 建立基本的邻接矩阵等
    adjs, d_es = paser.main(save=True)
    a = secondParse(adjs, d_es)
    # 分割子图
    # print(a.ds)
    sus = a.main()
    # 求feature
    a.third()
    ######################
    # 调试graph_onebyone
    #parser = graphParser()
    # adjs, d_es = parser.main(save=True)
    #d_gns = parser.ex_which_graph()
    #print(d_gns)
