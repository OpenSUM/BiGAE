import torch
import json
import argparse

import numpy as np
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt

import networkx as nx
import networkx.algorithms.centrality as nx_cen

import copy

RUN = 'edge'

edge_nx_cen_metric = [
    # nx_cen.edge_load_centrality
    nx_cen.edge_betweenness_centrality
    # nx_cen.dispersion
    # nx_cen.edge_betweenness_centrality
]
node_nx_cen_metric = [
    nx_cen.degree_centrality,
    # nx_cen.eigenvector_centrality,
    nx_cen.katz_centrality,
    nx_cen.closeness_centrality,
    # nx_cen.betweenness_centrality,
    nx_cen.load_centrality,
    # nx_cen.harmonic_centrality,
    # nx_cen.betweenness_centrality_source
    # nx_cen.voterank,
    # nx_cen.communicability_betweenness_centrality, RuntimeWarning: invalid value encountered in true_divide, 会卡死
    # nx_cen.group_betweenness_centrality, division by zero
    # nx_cen.current_flow_closeness_centrality, need to be connected
    # nx_cen.current_flow_betweenness_centrality, Graph not connected.
    # nx_cen.global_reaching_centrality, 'float' object has no attribute 'values'
    # nx_cen.percolation_centrality, ?
    # nx_cen.second_order_centrality, Non connected graph.
    # nx_cen.trophic_levels, not implemented for undirected type
]
if RUN == 'node':
    nx_cen_metric = node_nx_cen_metric
else:
    nx_cen_metric = edge_nx_cen_metric


def merge_json_pt(json_file, pt_file, new_file=None):
    ret = json.load(open(json_file, "r", encoding="utf8"))

    data = torch.load(pt_file)
    for i in range(len(ret)):
        ret[i]["labels"] = data[i]["labels"]
    ret = [x for x in ret if len(x["labels"]) > 0]
    print("after", len(ret))
    if new_file is not None:
        json.dump(ret, open(new_file, "w"))
    else:
        return ret


def show_graph(data):
    words = data['src'].split(" ")

    nodes = []
    i = 0
    for node in data['nodes']:
        cnode = ["%d" % i]
        i += 1
        for id in node:
            cnode.append(words[id])
        # nodes.append("".join(cnode).replace("Ġ", " "))
        nodes.append(" ".join(cnode).replace(" ##", ""))

    G = nx.Graph()
    # G.add_nodes_from(nodes)

    for edge in data['edges']:
        u, v = edge
        if u >= len(nodes) or v >= len(nodes):
            continue
        G.add_edge(nodes[u], nodes[v])

    nx.draw(G, with_labels=True)  # , pos=nx.circular_layout(G))
    # plt.savefig("./test2.png")
    plt.show()
    plt.close()


cnt = 0
fcnt = 0


def get_importance(data, key):  # key = merge_graph
    global cnt, fcnt
    G = nx.Graph()
    # nodes = data['nodes']
    # edges = [e for e in data['edges'] if e[0] < len(nodes) and e[1] < len(nodes)]
    # G.add_edges_from(edges)
    edges = copy.deepcopy(data[key])  # ['graph']
    for x in edges:
        x[0] += 512

    # for x in data[key]:
    #    edges.append([x[1], x[0]])
    # for i in range(len(src_ids)):
    #    edges.append([i,i])

    src_ids = data['src'][:512]

    G.add_edges_from(edges)
    scores = []

    f = 0
    len_score = -1
    for metric in nx_cen_metric:
        try:
            # print(metric)
            dic = {}
            score = None
            if metric == nx_cen.group_betweenness_centrality:
                score = metric(G, G.nodes)
            elif metric == nx_cen.voterank:
                len_nodes = len(src_ids)  # len_score
                score = metric(G, number_of_nodes=len_nodes)
                i = len(score)
                for idx in score:
                    dic[idx] = i
                    i -= 1
                score = []
                for i in range(len_nodes):
                    if i in dic.keys():
                        score.append(dic[i])
                    else:
                        score.append(0)
            elif metric == nx_cen.katz_centrality:
                score = nx_cen.katz_centrality_numpy(G)  # , max_iter=5000)
                score = list(score.values())
            elif metric == nx_cen.edge_betweenness_centrality:
                score = metric(G, normalized=True)  # , max_iter=5000)
                score = [max(score.get(tuple(edge), 0), score.get(tuple([edge[1], edge[0]]), 0)) for edge in edges]
            elif metric == nx_cen.edge_load_centrality:
                score = metric(G)  # , max_iter=5000)
                score_tmp = sorted(score.items(), key=(lambda x: (x[0], x[1])), reverse=False)
                score = dict(score_tmp)
                # print("Score", len(score), score, edges)
                score = list(score.values())
            else:
                score = metric(G)
                score = list(score.values())
            scores.append(score)
            """
            if len_score == -1:
                len_score = len(score)
            elif len_score != len(score):
                print(metric, len_score, len(score))
                print(dic, score)
            """
        except Exception as e:
            print("error")
            print(score, metric)
            print(e)
            f = 1
    if f:
        fcnt += 1
        scores = None
    else:
        cnt += 1
    print(cnt, fcnt)
    return scores


def discretize_feature(feature, num_bin):
    feature = np.array(feature)
    est = preprocessing.KBinsDiscretizer(n_bins=[num_bin for i in range(feature.shape[1])], encode='ordinal')
    est.fit(feature)
    feature = est.transform(feature)
    return feature


def get_discrete_importance(data, importance_dim, key):
    nums_node = []
    fea = []
    new_data = []

    succeed_bool = []
    for x in data:
        if x[key] == None:
            succeed_bool.append(0)
            print("continue")
            continue
        imp = get_importance(x, key)
        if imp is not None:
            fea.append(imp)
            new_data.append(x)
            nums_node.append(len(imp[0]))
            succeed_bool.append(1)
        else:
            succeed_bool.append(0)
    print("finish calculate centrality, begin discretize")
    tmp = [[] for i in range(len(fea[0]))]
    for i in range(len(fea)):
        for j in range(len(fea[0])):
            tmp[j].extend(fea[i][j])

    # tmp is feature list, size: cen_dim * num_nodes
    # print(np.array(tmp))
    fea_continu = np.array(tmp).transpose(1, 0)
    fea_discr = discretize_feature(fea_continu, importance_dim).astype(int)
    feature = []
    sum_len = 0
    for i in range(len(nums_node)):
        feature.append(fea_discr[sum_len:sum_len + nums_node[i]].tolist())
        sum_len += nums_node[i]

    return feature, succeed_bool  # new_data


def get_con_importance(data, importance_dim, key):
    nums_node = []
    fea = []
    new_data = []

    succeed_bool = []
    for x in data:
        if x[key] == None:
            succeed_bool.append(0)
            print("continue")
            continue
        imp = get_importance(x, key)
        if imp is not None:
            fea.append(imp)
            new_data.append(x)
            nums_node.append(len(imp[0]))
            succeed_bool.append(1)
        else:
            succeed_bool.append(0)

    return fea, succeed_bool


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_path", default='cnn_graph', type=str)
    parser.add_argument("-save_path", default='dis_small.json')
    parser.add_argument("-cen_dim", default=10, type=int)
    parser.add_argument("-file_id", default=0, type=int)  # 起始的ID
    parser.add_argument("-block_size", default=19, type=int)
    parser.add_argument("-test_file_num", default=0, type=int)
    args = parser.parse_args()

    datas = []
    lens_train = [0]
    dir_in = args.input_path
    dir_out = args.save_path
    graph_key = 'merge_graph'
    file_set_id = args.file_id  # 0#7
    block_size = args.block_size  # 75#20

    if file_set_id * block_size > 19: quit()
    if (file_set_id + 1) * block_size >= 19:
        print("test_file_num added")
        test_file_num = 3
    else:
        test_file_num = 0

    for i in range(file_set_id * block_size, min(19, (file_set_id + 1) * block_size)):
        # data = torch.load('%s/cnndaily.train.%d.pt'%(dir_in, i))
        data = torch.load('%s/nyt.train.%d.pt' % (dir_in, i))
        # data = torch.load('%s/multinews.train.%d.pt'%(dir_in, i))
        # data = [{'src':d['src'], 'src_txt':d['src_txt'], graph_key:d[graph_key]} for d in data]
        # data = data[0:10]
        new_len = lens_train[-1] + len(data)
        lens_train.append(new_len)
        datas.extend(data)
        print("train_file", i)
    lens_test = [lens_train[-1]]
    for i in range(test_file_num):
        # data = torch.load('%s/cnndaily.test.%d.pt'%(dir_in, i))
        data = torch.load('%s/nyt.test.%d.pt' % (dir_in, i))
        # data = torch.load('%s/multinews.test.%d.pt'%(dir_in, i))
        # data = [{'src':d['src'], 'src_txt':d['src_txt'], graph_key:d[graph_key]} for d in data]
        new_len = lens_test[-1] + len(data)
        lens_test.append(new_len)
        datas.extend(data)

    print("begin")
    # datas = datas[:10]
    feature, succeed_bool = get_con_importance(datas, args.cen_dim, graph_key)
    print("end")

    j = 0
    for i in range(len(datas)):
        if succeed_bool[i]:
            datas[i]['between_edge_attr'] = feature[j][0]  # 和gen_graph中得到的merge_edge_attr有何区别？
            j += 1

    print(len(datas))
    print("len_train", lens_train, "len_test", lens_test)
    for i in range(0, len(lens_train) - 1):
        data = datas[lens_train[i]:lens_train[i + 1]]
        # torch.save(data, '%s/cnndaily.train.%d.pt'%(dir_out, file_set_id*block_size+i))
        torch.save(data, '%s/nyt.train.%d.pt' % (dir_out, file_set_id * block_size + i))
        # torch.save(data, '%s/multinews.train.%d.pt'%(dir_out, file_set_id*block_size+i))

    for i in range(len(lens_test) - 1):
        data = datas[lens_test[i]:lens_test[i + 1]]
        # torch.save(data, '%s/cnndaily.test.%d.pt'%(dir_out, i))
        torch.save(data, '%s/nyt.test.%d.pt' % (dir_out, i))
        # torch.save(data, '%s/multinews.test.%d.pt'%(dir_out, i))

    """
    data = json.load(open(args.json_path, "r", encoding="utf8"))
    #data = merge_json_pt(args.json_path, args.pt_path)
    feature,data = get_discrete_importance(data, args.cen_dim)
    for i in range(len(data)):
        data[i]['importance'] = feature[i]
    json.dump(data, open(args.save_path, 'w'), indent=2)

    #show_graph(data[0])
    """


