import torch
import glob
import random 
import bisect
from torch.utils.data import Dataset

import torch_geometric as pyg

NUM_METRIC = 4
def pad(data, pad_id, width=-1):
    if (width == -1):
        width = max(len(d) for d in data)
    rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
    return rtn_data

def map_imp(imp):
    imp = min(imp*50, 1)
    return imp #imp/2 + 0.5

def get_graph_data(src, clss, pre_src, pre_graphs, pre_clss, pre_between_edge_attr, batch_size, device):
    #sent_importance = pad(pre_sent_importance, 0)
    graphs = []
    graphs_centrality = []
    summ_graphs = []
    sent_len = len(src[0])
    for bi in range(batch_size):
        n_node = len(pre_src[bi]) # (pre_src_sent_labels[bi])

        #graph = [[e[0]+sent_len, e[1]] for e in pre_graphs[bi] if e[0] < n_node and e[1] < n_node]
        #graph.extend([[e[1], e[0]+sent_len] for e in pre_graphs[bi] if e[0] < n_node and e[1] < n_node])
        graph = []
        graph_centrality = []
        for i, e in enumerate(pre_graphs[bi]):
            if e[0] < n_node and e[1] < n_node:
                graph.append([e[0]+sent_len, e[1]])
                graph.append([e[1], e[0]+sent_len])
                graph_centrality.append(map_imp(pre_between_edge_attr[bi][i]))
                graph_centrality.append(map_imp(pre_between_edge_attr[bi][i]))
        graphs.append(graph)
        graphs_centrality.append(graph_centrality)

        summ_graph = [e for e in graph if e[0]-sent_len in pre_clss[bi] or e[1]-sent_len in pre_clss[bi] ]
        summ_graphs.append(summ_graph)

    data_list = [pyg.data.Data(num_nodes=len(src[0])+len(clss[0]), \
            centrality=torch.tensor(graphs_centrality[i]), \
            summ_edge_index=torch.tensor(summ_graphs[i]).t().contiguous(), \
            edge_index=torch.tensor(graphs[i]).t().contiguous() ) \
            for i in range(len(graphs))]
            
    return pyg.data.Batch.from_data_list(data_list).to(device)

#输入到dataloader里的一个函数，作用是把一个batch里的多条数据合到一起。对中心性的东西也要做一下
def gae_collate_fn(data):
    device = data[0][-1]
    is_test = data[0][-2]
    ret = {}
    if data is not None:
        batch_size = len(data)
        pre_src = [x[0] for x in data]
        pre_tgt = [x[1] for x in data]
        pre_clss = [x[2] for x in data]
        pre_src_sent_labels = [x[3] for x in data]
        pre_graphs  = [x[4] for x in data]
        pre_between_edge_attr = [x[5] for x in data]
        pre_src_for_sent = [x[6] for x in data]

        src = torch.tensor(pad(pre_src, 0))
        tgt = torch.tensor(pad(pre_tgt, 0))
        maxd = 0 
        for sents in pre_src_for_sent:
            d = max(len(x) for x in sents)
            maxd = max(d, maxd)
        src_for_sent = [pad(x, 0, maxd) for x in pre_src_for_sent]
        src_for_sent = torch.tensor(pad(src_for_sent, [0]*maxd))
        ret["src_for_sent"] = src_for_sent.to(device)

        mask_src = 1 - (src == 0).int()
        mask_tgt = 1 - (tgt == 0).int()

        clss = torch.tensor(pad(pre_clss, -1))
        src_sent_labels = torch.tensor(pad(pre_src_sent_labels, 0))
        mask_cls = 1 - (clss == -1).int()
        clss[clss == -1] = 0
        ret['clss'] = clss.long().to(device)
        ret['mask_cls'] = mask_cls.to(device)
        ret['labels'] = src_sent_labels.to(device)

        ret['src'] = src.to(device)
        ret['tgt'] = tgt.to(device)
        ret['mask_src'] = mask_src.to(device)
        ret['mask_tgt'] = mask_tgt.to(device)

        ret['graph_data'] = get_graph_data(src, clss, pre_src, pre_graphs, pre_clss, pre_between_edge_attr, batch_size, device)

        if (is_test):
            src_str = [x[-4] for x in data]
            ret['src_str'] = src_str
            tgt_str = [x[-3] for x in data]
            ret['tgt_str'] = tgt_str
    return ret


class GAEDataset(Dataset):
    def __init__(self, args, device, shuffle=True, is_test=False):
        self.args = args
        self.batch_size = args.batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = (args.mode != "train")
        self.dataset = []
        self.load_dataset(args.mode)
        self.preprocessed = []
    
    def load_dataset(self, corpus_type):
        assert corpus_type in ["train", "valid", "test", "test_ae"]
        if corpus_type == "test_ae": corpus_type = "test"
    
        # Sort the glob output by file name (by increasing indexes).
        pts = sorted(glob.glob(self.args.data_path + '.' + corpus_type + '.[0-9]*.pt'))
        if pts:
            if (self.shuffle):
                random.shuffle(pts)
    
            for pt in pts:
                dataset = torch.load(pt)
                print('Loading %s dataset from %s, number of examples: %d' %
                        (corpus_type, pt, len(dataset)))
                self.dataset.extend(dataset)
        else:
            # Only one inputters.*Dataset, simple!
            pt = self.args.data_path + '.' + corpus_type + '.pt'
            self.dataset = torch.load(pt)

    def preprocess(self, ex):
        src = ex['src']
        tgt = ex['tgt'][:self.args.max_tgt_len]
        src_sent_labels = ex['src_sent_labels']
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']
        graph = ex['merge_graph']
        # importance = ex['merge_edge_attr']# ex['graph_importance_con'] #TODO: 替换成centrality
        centrality = ex['between_edge_attr']

        end_id = [src[-1]]
        src = src[:-1][:self.args.max_pos - 1] + end_id
        max_sent_id = bisect.bisect_left(clss, self.args.max_pos)
        src_sent_labels = src_sent_labels[:max_sent_id]
        clss = clss[:max_sent_id]
        src_for_sent = self.get_src_for_sentence(src, clss)
        # 处理centrality
        centrality = centrality[:self.args.max_pos]
        # importance = importance[:self.args.max_pos]#[max_sent_id]

        if(self.is_test):
            return [src, tgt, clss, src_sent_labels, graph, centrality, src_for_sent, src_txt, tgt_txt]
        else:
            return [src, tgt, clss, src_sent_labels, graph, centrality, src_for_sent]
    
    def get_src_for_sentence(self, src, clss):
        tmp_clss = clss.copy()
        tmp_clss.append(len(src))
        src_for_sent = [(src[tmp_clss[j]:tmp_clss[j+1]])[:self.args.max_sent_len] \
            for j in range(len(clss))]
        return src_for_sent

    def __getitem__(self, index):
        ret = []
        if index in self.preprocessed:
            ret = self.dataset[index]
        else:
            ret = self.preprocess(self.dataset[index])
            self.dataset[index] = ret
            self.preprocessed.append(index)
        ret.extend([self.is_test, self.device])
        return ret

    def __len__(self):
        return len(self.dataset)
