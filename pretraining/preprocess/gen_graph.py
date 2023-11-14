#from allennlp.predictors.predictor import Predictor
#import allennlp_models.tagging
import re
import nltk
import argparse
import json
import time
import numpy as np
import torch

from nltk.corpus import stopwords
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from embedding import Vocab

FILTERWORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``',
                '-', '--', '|', '\/']
FILTERWORD.extend(punctuations)
filter_set = set(FILTERWORD)

def get_all_entity(sent):
    ret = nltk.word_tokenize(sent)
    ret = set(ret)- filter_set 
    return ret

def get_filterword_id(vocab):
    filter_list = []
    word_list = vocab.word_list()
    for w in FILTERWORD:
        if w in word_list:
            filter_list.append(vocab.word2id(w))
    return filter_list



# sentence_list = nltk.sent_tokenize(article)
def get_sentence_overlap(sentence_list):
    adj_array = []
    no_ent = 0
    docs = []
    t1 = time.time()
    for sen1 in sentence_list:
        #print("out1", sen1)
        doc = get_all_entity(sen1)
        #print("out2", doc)
        docs.append(doc)
    edges = []
    for i, sen1 in enumerate(sentence_list):
        doc1 = docs[i]
        ents1 = set(i for i in doc1)
        if len(ents1) == 0:
            no_ent += 1 
            #print(sentence_list[i])
        adj = []
        for j,sen2 in enumerate(sentence_list):
            doc2 = docs[j]#get_all_entity(sen2)
            ents2 = set(i for i in doc2)
            ent_overlap = ents1 & ents2
            if len(ent_overlap) > 0:
                edges.append([i,j])
    t2 = time.time()

    return edges

def get_1gram_graph():
    for i in range(6):
        #data = json.load(open("out_dir/test%i.json"%d, "r", encoding="utf8"))
        adata = torch.load('src_knn_gen/knn_graph.test.%d.pt'%i)
        ndata = []
        avg_ratio = 0
        for j in range(len(adata)):
            #try:
            adj = get_sentence_overlap(adata[j]['src_txt'])
            #print("adj",adj)
            #except:
            #    adj = None
            #print(len(adj), len(adata[j]['src_txt']), len(adj)/len(adata[j]['src_txt']))
            print( len(adj)/len(adata[j]['src_txt'])**2 )
            ratio = len(adj)/len(adata[j]['src_txt'])**2 
            avg_ratio += ratio
            if adj != None:
                adata[j]['overlap_graph'] = adj
                ndata.append(adata[j])
        print(avg_ratio/len(adata))
        quit()
        torch.save(ndata, 'knn_input_tmp/cnndaily.test.%d.pt'%i)

    
    for i in range(143):
        adata = torch.load('src_knn_gen/knn_graph.train.%d.pt'%i)
        ndata = []
        for j in range(len(adata)):
            #try:
            adj = get_sentence_overlap(adata[j]['src_txt'])
            #except:
            #    adj = None
            if adj != None:
                adata[j]['overlap_graph'] = adj
                ndata.append(adata[j])
        torch.save(ndata, 'knn_input_tmp/cnndaily.train.%d.pt'%i)

vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
def get_tfidf(data):
    clss = data['clss']+[len(data['src'])]
    num_sen = len(clss)-1
    corpus0 = [data['src'][clss[i]:clss[i+1]] for i in range(num_sen)]

    corpus = [""]*num_sen
    for i in range(num_sen):
        corpus[i] = " ".join(["x%d"%i for i in corpus0[i]])

    X = vectorizer.fit_transform(corpus)
    tfidf=transformer.fit_transform(X)#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
    weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    word_dict = {word[i]:i for i in range(len(word))}
    ret = []
    for i in range(num_sen):
        for j,x in enumerate(corpus0[i]):
            #print(word[j], weight[i][word_dict["x%d"%x]])
            ret.append(weight[i][word_dict["x%d"%x]])
    return ret

vocab_size = 600000
first_position = [-1] * vocab_size
last_position = [-1] * vocab_size 
CLS_ID = 101
#2 for albert, 101 for bert

def get_merge_graph(data, filter_vocab_id):
    global first_position
    src_id = data['src'][:512]
    clss = data['clss'].copy()
    clss.append(len(src_id))
    for i,x in enumerate(src_id):
        if first_position[x] == -1:
            first_position[x] = i
    
    edges  = []
    edge_attr = []
    tfidf = get_tfidf(data)
    for i in range(len(clss)-1):
        if clss[i] >= 512: break
        for j in range(clss[i], min(512,clss[i+1])):
            word_id = src_id[j]
            if word_id in filter_vocab_id: continue
            if [i, first_position[word_id]] in edges: continue
            edges.append([i, first_position[word_id]])
            edge_attr.append(tfidf[j])

    for i,x in enumerate(src_id):
            first_position[x] = -1

    return edges, edge_attr


def get_split_graph(data, filter_vocab_id):
    global first_position#, last_position
    src_id = data['src'][:512]
    word_cls = [[] for i in range(len(src_id))]
    for i,x in enumerate(src_id):
        if first_position[x] == -1:
            first_position[x] = i
        word_cls[first_position[x]].append(i)
    
    edges  = []
    edge_attr = []
    tfidf = get_tfidf(data)
    for i in range(len(src_id)):
        if src_id[i] == CLS_ID:
            for j in range(i+1, len(src_id)):
                word_id = src_id[j]
                if word_id == CLS_ID: break
                if word_id in filter_vocab_id: continue
                word_cls_id = first_position[word_id]
                if [i, word_cls[word_cls_id][0]] in edges: continue
                edge_factor = 1#len(word_cls[word_cls_id])
                for k in word_cls[word_cls_id]:
                    edges.append([i, k])
                    edge_attr.append(tfidf[j]/edge_factor)

    for i,x in enumerate(src_id):
            first_position[x] = -1
    return edges, edge_attr

def tmp_change_src(data):
    tmp = []
    for x in data['src']: tmp.append(x)
    data['src'] = tmp
    return data


def get_heter_graph(args, vocab):
    filter_vocab_id = get_filterword_id(vocab)
    in_dir = args.input_path
    out_dir = args.save_path
    file_num = 19
    for i in range(args.test_file_num):
        adata = torch.load('%s/nyt.test.%d.bert.pt'%(in_dir, i))
        #adata = torch.load('%s/multinews.test.%d.bert.pt'%(in_dir, i))
        #adata = torch.load('%s/cnndm_bert.pt.test.%d.bert.pt'%(in_dir, i))
        #adata = torch.load('%s/cnndaily.test.%d.pt'%(in_dir, i))
        #ndata = []
        avg_ratio = 0
        for j in range(len(adata)):
            adata[j] = tmp_change_src(adata[j])
            #split_graph, split_edge_attr = get_split_graph(adata[j])
            merge_graph, merge_edge_attr = get_merge_graph(adata[j], filter_vocab_id)
            #adata[j]['split_graph'] = split_graph
            adata[j]['merge_graph'] = merge_graph
            adata[j]['merge_edge_attr'] = merge_edge_attr
            #adata[j]['split_edge_attr'] = split_edge_attr
            #print("test", j, adata[j])
        torch.save(adata, '%s/nyt.test.%d.pt'%(out_dir, i))
        #torch.save(adata, '%s/multinews.test.%d.pt'%(out_dir, i))
        #torch.save(adata, '%s/cnndaily.test.%d.pt'%(out_dir, i))

    
    for i in range(args.file_id*args.block_size, min(file_num, (args.file_id+1)*args.block_size) ):
        adata = torch.load('%s/nyt.train.%d.bert.pt'%(in_dir, i))
        #adata = torch.load('%s/multinews.train.%d.bert.pt'%(in_dir, i))
        #adata = torch.load('%s/cnndm_bert.pt.train.%d.bert.pt'%(in_dir, i))
        #adata = torch.load('%s/cnndaily.train.%d.pt'%(in_dir, i))
        #ndata = []
        for j in range(len(adata)):
            adata[j] = tmp_change_src(adata[j])
            #split_graph, split_edge_attr = get_split_graph(adata[j])
            merge_graph, merge_edge_attr = get_merge_graph(adata[j], filter_vocab_id)
            #adata[j]['split_graph'] = split_graph
            adata[j]['merge_graph'] = merge_graph
            adata[j]['merge_edge_attr'] = merge_edge_attr
            #adata[j]['split_edge_attr'] = split_edge_attr
            print("train", j)
        torch.save(adata, '%s/nyt.train.%d.pt'%(out_dir, i))
        #torch.save(adata, '%s/multinews.train.%d.pt'%(out_dir, i))
        #torch.save(adata, '%s/cnndaily.train.%d.pt'%(out_dir, i))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_path", default='out_dir_disc_trunc')
    parser.add_argument("-save_path", default='out_tfidf')
    parser.add_argument("-file_id", default=0, type=int)
    parser.add_argument("-block_size", default=143, type=int)
    parser.add_argument("-test_file_num", default=0, type=int)
    parser.add_argument("-vocab_path", default='/home/LAB/projects/AutoEncoder/gignore/vocab.txt')
    parser.add_argument("-vocab_size", default=50000, type=int)
    args = parser.parse_args()
    vocab = Vocab(args.vocab_path, args.vocab_size)
    print(args)
    get_heter_graph(args, vocab)


