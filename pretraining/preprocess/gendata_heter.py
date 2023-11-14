import torch
import json
import argparse
import gc

import numpy as np
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt

import pickle
import networkx as nx
import networkx.algorithms.centrality as nx_cen

def simple_discretize_edge_feature(data):
    for i in range(len(data)):
        attr = data[i]['between_edge_attr']
        data[i]['between_edge_attr'] = [min(9, int(i*1000)) for i in attr]
        attr = data[i]['split_edge_attr']
        data[i]['split_edge_attr'] = [min(9, int(i*20)) for i in attr]
    return data

def select_4(datas):
    for i in range(len(datas)):
        datas[i]['graph_importance_con'] = [datas[i]['graph_importance_con'][j] for j in [0,1,3,5]]
    return datas
    """
    feature = []
    for i in range(len(datas)):
        fea = datas[i]['merge_graph_importance_con']
        feature.append([fea[0], fea[1], fea[3], fea[5]])
    print(len(feature))
    """

def trunc_data(datas):
    max_src_len = 512
    #max_edge_len = 512
    for i in range(len(datas)):
        if 'graph_importance_con' in datas[i].keys():
            datas[i]['importance'] = datas[i]['graph_importance_con'][:max_src_len].copy()
            edges = []
            tfidf = []
            edge_attr = []
            for j,e in enumerate(datas[i]['split_graph']):
                if e[0] < max_src_len and e[1] < max_src_len:
                    edges.append(e)
            datas[i]['split_graph'] = edges
            datas[i].pop('graph_importance_con')
        else:
            print("L51  graph_importance_con not in keys")
    return datas


def discretize_feature(feature, num_bin, discre_path):
    feature = np.array(feature)
    est = preprocessing.KBinsDiscretizer(n_bins=[num_bin for i in range(feature.shape[1])], encode='ordinal')
    est.fit(feature)
    feature = est.transform(feature)
    #with open("discretizer.pickle", 'wb') as f:
    with open(discre_path, 'wb') as f:
        pickle.dump(est, f)
    return feature

def convert_edge_cen(datas):
    edge_attr = []
    node_attr = []
    n_datas = []
    for i,d in enumerate(datas):
        #print(d['graph_importance_con'])
        try:
            node_attr = d['graph_importance_con'][0]
            for e in d['split_graph']:
                print(node_attr[e[0]],e[0], e[1])
                edge_attr.append(node_attr[e[0]][e[1]])
        except Exception as e:
            print(e)
        n_datas.append({'src':d['src'],'cen_edge_attr':edge_attr})
    del datas

    #discretize
    discretize_feature(edge_attr, args.cen_dim, 'edge_discretizer.pickle')
    return n_datas


def merge(data1, data2, data3):
    #full tfidf node
    """
    for i in range(len(data3)):
        data1[i]['split_graph'] = data2[i]['split_graph']
        data1[i]['split_tfidf'] = data2[i]['split_edge_attr']

    for i in range(len(data3)):
        data1[i]['importance'] = data3[i]['graph_importance_con']
    return data1
    """
    j = 0
    for i in range(len(data1)):
        data1[i]['src_sent_labels0'] = data1[i]['src_sent_labels']
        data1[i].pop('src_sent_labels')
        if j >= len(data2): break
        if len(data1[i]['clss']) == len(data2[j]['clss']):
            data1[i]['src_sent_labels'] = data2[j]['src_sent_labels']
            j += 1
        else:
            if len(data1[i]['clss']) == len(data2[j+1]['clss']):
                j += 1
                if j >= len(data2): break
                data1[i]['src_sent_labels'] = data2[j]['src_sent_labels']
                j += 1
            elif len(data1[i+1]['clss']) != len(data2[j]['clss']):
                print("error i j:", i, j)

    print("final", len(data1), len(data2), j)
    return data1


def read_data(args, dir_in, graph_key=None, file_num=143):
    datas = []
    lens_train = [0]
    file_set_id = args.file_id
    block_size = args.block_size
    test_file_num = args.test_file_num

    for i in range(file_set_id*block_size, min(file_num,(file_set_id+1)*block_size)):
        data = torch.load('%s/cnndaily.train.%d.pt'%(dir_in, i))
        #for j in range(len(data)):    gc.collect()
        new_len = lens_train[-1]+len(data)
        lens_train.append(new_len)
        datas.extend(data)
        print("train_file", i, len(data))
        for j in range(0, 9):
            print(len(data[j]['clss']), end=" ")
        print()
        for j in range(1, 10):
            print(len(data[-j]['clss']), end=" ")
        print()

    lens_test = [lens_train[-1]]
    for i in range(test_file_num):
        data = torch.load('%s/cnndaily.test.%d.pt'%(dir_in, i))
        #data = [{'src':d['src'], 'src_txt':d['src_txt'], graph_key:d[graph_key]} for d in data]
        new_len = lens_test[-1]+len(data)
        lens_test.append(new_len)
        datas.extend(data)
        print("test_file", i, len(data))
    return datas, lens_train, lens_test


def save_data(args, dir_out, datas, lens_train, lens_test):
    file_set_id = args.file_id#0#7
    print(file_set_id*args.block_size)
    #print(len(datas))
    print("len_train", lens_train, "len_test", lens_test)

    for i in range(0, len(lens_train)-1):
        data = datas[lens_train[i]:lens_train[i+1]]
        ndata = []
        for j in range(len(data)):
            if 'importance' in data[j].keys() and 'src_sent_labels' in data[i].keys():
                ndata.append(data[j])
        print("\n", j)
        for j in range(0, 9):
            print(len(ndata[j]['clss']), len(ndata[j]['src_sent_labels']), end=" ")
        data = ndata
        torch.save(ndata, '%s/cnndaily.train.%d.pt'%(dir_out, (file_set_id*args.block_size)+i))

    for i in range(len(lens_test)-1):
        data = datas[lens_test[i]:lens_test[i+1]]
        ndata = []
        for j in range(len(data)):
            if 'importance' in data[j].keys() and 'src_sent_labels' in data[i].keys():
                ndata.append(data[j])
        data = ndata
        torch.save(ndata, '%s/cnndaily.test.%d.pt'%(dir_out, i))


def main_merge(args):
    dir_out = 'bert_2sen'
    graph_key = 'split_graph'#'merge_graph'

    file_set_id = args.file_id#0#7
    if file_set_id*args.block_size > 143: quit()

    print("begin")
    print(args)

    #datas = select_4(datas)
    data2, lens_train, lens_test = read_data(args, '../old_PreSumm/PreSumm-red/bert_cnn_2sen/', graph_key, 144)
    print("read full data")
    data1, lens_train, lens_test = read_data(args, 'out_dir_disc_trunc', graph_key)
    print("read tfidf data")
    #assert len(data1)==len(data2)
    print(len(data1), len(data2))
    
    datas = merge(data1, data2, None)

    save_data(args, dir_out, datas, lens_train, lens_test)
    print("finish")


def main_edge_attr(args):
    dir_out = 'out_dir_edge_attr'
    graph_key = 'split_graph'#'merge_graph'

    file_set_id = args.file_id#0#7
    if file_set_id*args.block_size > 143: quit()

    print("begin")
    print(args)

    #datas = select_4(datas)
    data, lens_train, lens_test = read_data(args, 'heter_cen_con_edge', graph_key)
    datas = convert_edge_cen(data)

    save_data(args, dir_out, datas, lens_train, lens_test)
    print("finish")


def main_sel4(args):
    dir_in = 'heter_cen_con_node'
    dir_out = 'out_dir_n'
    graph_key = 'split_graph'

    file_set_id = args.file_id
    if file_set_id*args.block_size > 143: quit()

    print("begin")
    print(args)
    datas, lens_train, lens_test = read_data(args, dir_in, graph_key)

    datas = select_4(datas)
    save_data(args, dir_out, datas, lens_train, lens_test)
    print("finish")


def main_clean(args):
    dir_in = 'albert_disc'
    dir_out = 'albert_clean2'
    graph_key = 'split_graph'

    file_set_id = args.file_id
    if file_set_id*args.block_size > 143: quit()

    print("begin")
    print(args)
    datas, lens_train, lens_test = read_data(args, dir_in, graph_key)

    datas = trunc_data(datas)
    save_data(args, dir_out, datas, lens_train, lens_test)
    print("finish")

def main_simple_disc(args):
    dir_in = 'out_2attr_edge'
    dir_out = 'out_2attr_disc'

    file_set_id = args.file_id
    if file_set_id*args.block_size > 143: quit()

    print("begin")
    print(args)
    datas, lens_train, lens_test = read_data(args, dir_in)

    datas = simple_discretize_edge_feature(datas)
    save_data(args, dir_out, datas, lens_train, lens_test)
    print("finish")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-save_path", default='dis_small.json')
    parser.add_argument("-cen_dim", default=10, type=int)
    parser.add_argument("-file_id", default=0, type=int)
    parser.add_argument("-block_size", default=143, type=int)
    parser.add_argument("-test_file_num", default=0, type=int)
    parser.add_argument("-task", default='merge', type=str)
    args = parser.parse_args()
    print(args)

    if args.task == 'merge':
        main_merge(args)
    elif args.task == 'sel4':
        main_sel4(args)
    elif args.task == 'edge':
        main_edge_attr(args)
    elif args.task == 'clean':
        main_clean(args)
    elif args.task == 'sim_disc':
        main_simple_disc(args)

