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

def discretize_feature(feature, num_bin, discre_path):
    feature = np.array(feature)
    est = preprocessing.KBinsDiscretizer(n_bins=[num_bin for i in range(feature.shape[1])], encode='ordinal')
    est.fit(feature)
    feature = est.transform(feature)
    #with open("discretizer.pickle", 'wb') as f:
    with open(discre_path, 'wb') as f:
        pickle.dump(est, f)
    return feature

def load_and_discretize_feature(feature, num_bin, discre_path):
    with open(discre_path, 'rb') as f:
        est = pickle.load(f)
    est.fit(feature)
    feature = est.transform(feature)
    #with open("discretizer.pickle", 'wb') as f:
    return feature

def get_discrete_importance(args, fea, importance_dim, discre_path):
    nums_node = []
    new_data = []

    for x in fea:
        nums_node.append(len(x[0]))
    tmp = [[] for i in range(len(fea[0]))]
    for i in range(len(fea)):
        if(len(fea[0])!=len(fea[1])): print("get_discrete_importace L41", i,0,1)
        if(len(fea[0])!=len(fea[2])): print("get_discrete_importace L42", i,0,2)
        if(len(fea[0])!=len(fea[3])): print("get_discrete_importace L43", i,0,3)
        for j in range(len(fea[0])):
            tmp[j].extend(fea[i][j])

    fea_continu = np.array(tmp).transpose(1, 0)
    if args.task == 'discretizer':
        fea_discr = discretize_feature(fea_continu, importance_dim, discre_path).astype(int)
    else:
        fea_discr = load_and_discretize_feature(fea_continu, importance_dim, discre_path).astype(int)
    feature = []
    sum_len = 0
    for i in range(len(nums_node)):
        feature.append(fea_discr[sum_len:sum_len+nums_node[i]].tolist())
        sum_len += nums_node[i]
    
    return feature


def read_data(args, dir_in, graph_key):
    datas = []
    lens_train = [0]
    file_set_id = args.file_id
    block_size = args.block_size
    test_file_num = args.test_file_num

    for i in range(file_set_id*block_size, min(143,(file_set_id+1)*block_size)):
        data = torch.load('%s/cnndaily.train.%d.pt'%(dir_in, i))
        #for j in range(len(data)):
        #    gc.collect()
        new_len = lens_train[-1]+len(data)
        lens_train.append(new_len)
        datas.extend(data)
        print("train_file", i)

    lens_test = [lens_train[-1]]
    for i in range(test_file_num):
        data = torch.load('%s/cnndaily.test.%d.pt'%(dir_in, i))
        #data = [{'src':d['src'], 'src_txt':d['src_txt'], graph_key:d[graph_key]} for d in data]
        new_len = lens_test[-1]+len(data)
        lens_test.append(new_len)
        datas.extend(data)
    return datas, lens_train, lens_test


def save_data(args, dir_out, datas, lens_train, lens_test):
    file_set_id = args.file_id#0#7
    print(len(datas))
    print("len_train", lens_train, "len_test", lens_test)

    for i in range(0, len(lens_train)-1):
        data = datas[lens_train[i]:lens_train[i+1]]
        torch.save(data, '%s/cnndaily.train.%d.pt'%(dir_out, file_set_id*args.block_size+i))
    for i in range(len(lens_test)-1):
        data = datas[lens_test[i]:lens_test[i+1]]
        torch.save(data, '%s/cnndaily.test.%d.pt'%(dir_out, i))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-save_path", default='dis_small.json')
    parser.add_argument("-cen_dim", default=10, type=int)
    parser.add_argument("-file_id", default=0, type=int)
    parser.add_argument("-block_size", default=143, type=int)
    parser.add_argument("-test_file_num", default=0, type=int)
    parser.add_argument("-task", default='data', type=str, choices=['data', 'discretizer'])
    args = parser.parse_args()

    fea_type = 'albert'#edge
    dir_out = 'albert_disc'
    key = 'graph_importance_con'
    dir_in = 'albert_cen_node'#'heter_cen_con_%s'%fea_type
    datas, lens_train, lens_test = read_data(args, dir_in, 'split_graph')

    feature = []
    err_num = 0
    for i in range(len(datas)):
        if key in datas[i].keys():
            fea = datas[i][key]
            feature.append(fea)#(([fea[0], fea[1], fea[3], fea[5]]))
        else:
            err_num += 1
            print("L123 error",err_num)
    print(len(feature))

    feature = get_discrete_importance(args, feature, args.cen_dim, '%s_discretizer.pickle'%fea_type)
    j = 0
    for i in range(len(datas)):
        if key in datas[i].keys():
            datas[i][key] = feature[j]
            j+=1

    """
    print(len(datas))
    print("len_train", lens_train, "len_test", lens_test)
    for i in range(0, len(lens_train)-1):
        data = datas[lens_train[i]:lens_train[i+1]]
        torch.save(data, '%s/cnndaily.train.%d.pt'%(dir_out, file_set_id*block_size+i))
    """
    save_data(args, dir_out, datas, lens_train, lens_test)


