import os
import re
from tokenize import group
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from gae import AutoEncoder
from utils import test_rouge, rouge_results_to_str
import collections
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


class Trainer():
    def __init__(self, args, model, embed=None):
        self.args = args
        self.model = model

    def test(self, test_iter, step, cal_lead=False, cal_oracle=False):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """

        # Set model in validating mode.
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False

        if (not cal_lead and not cal_oracle):
            self.model.eval()

        can_path = '%s_step%d.candidate' % (self.args.result_path, step)
        gold_path = '%s_step%d.gold' % (self.args.result_path, step)
        with open(can_path, 'w') as save_pred:
            with open(gold_path, 'w') as save_gold:
                with torch.no_grad():
                    for batch in test_iter:
                        batch_size = batch["src"].size(0)
                        labels = batch.pop("labels")
                        gold = []
                        pred = []

                        if (cal_lead): # Lead-3 选前三句话
                            selected_ids = [list(range(batch["clss"].size(1)))] * batch_size
                        elif (cal_oracle): # 选标准答案，计算标准答案的分数
                            selected_ids = [[j for j in range(batch["clss"].size(1)) if labels[i][j] == 1] for i in
                                            range(batch_size)]
                        else:
                            ret = self.model(**batch) #运行摘要模型
                            """
                            sent_scores = ret["sent_scores"] #从模型获取句子的分数
                            mask = batch["mask_cls"] #遮住padding的句子
                        
                            sent_scores = sent_scores + mask.float()
                            sent_scores = sent_scores.cpu().data.numpy()
                            selected_ids = np.argsort(-sent_scores, 1) #选择分数最高的句子
                            """
                            sent_vec = ret["sent_vec"] #获取句子表征
                            batch_num = sent_vec.size()[0]
                            selected_ids = np.zeros([sent_vec.size()[0],sent_vec.size()[1]])
                            for k in range(batch_num):
                                batch_vec = sent_vec[k]
                                size = batch_vec.size()[0]
                                sim_mat = np.zeros([size, size])

                                # textrank
                                """
                                for i in range(size):
                                    for j in range(size):
                                        if i != j:
                                            sim_mat[i][j] = torch.dot(batch_vec[i], batch_vec[j])
                                            #sim_mat[i][j] = torch.cosine_similarity(batch_vec[i].reshape(1,-1), batch_vec[j].reshape(1,-1))
                                nx_graph = nx.from_numpy_array(sim_mat)
                                scores = nx.pagerank(nx_graph)
                                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                                for l in range(size):
                                    selected_ids[k][l] = sorted_scores[l][0]
                                #"""

                                # lexrank
                                """
                                degrees = np.zeros(size)
                                for i in range(size):
                                    for j in range(size):
                                        sim_mat[i][j] = torch.cosine_similarity(batch_vec[i].reshape(1,-1), batch_vec[j].reshape(1,-1))
                                        if sim_mat[i][j] > 0.1:
                                            sim_mat[i][j] = 1.0
                                            degrees[i] += 1
                                        else: 
                                            sim_mat[i][j] = 0
                                for i in range(size):
                                    for j in range(size):
                                        if degrees[i] == 0:
                                            degrees[i] = 1
                                        
                                        sim_mat[i][j] = sim_mat[i][j] / degrees[i] 

                                transposed_sim_mat = sim_mat.T
                                p_vector = np.array([1.0 / size] * size)
                                lambda_val = 1.0

                                while lambda_val > 0.1:
                                    next_p = np.dot(transposed_sim_mat, p_vector)
                                    lambda_val = np.linalg.norm(np.subtract(next_p, p_vector))
                                    p_vector = next_p
                                
                                scores = []
                                for i in range(size):
                                    scores.append([i, p_vector[i]])

                                sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)     
                                for l in range(size):
                                    selected_ids[k][l] = sorted_scores[l][0]
                                #"""

                                # pacsum
                                """
                                encoder = np.zeros([size, size])
                                e_max = float("-inf")
                                e_min = float("inf")
                                for i in range(size):
                                    for j in range(size):
                                        if i != j:
                                            encoder[i][j] = torch.dot(batch_vec[i], batch_vec[j])
                                            e_max = max(encoder[i][j], e_max)
                                            e_min = min(encoder[i][j], e_min)
                                for i in range(size):
                                    for j in range(size):
                                        if i != j:
                                            sim_mat[i][j] = max(0, encoder[i][j] - (e_min + 0.05 * (e_max - e_min))) # cnn/dm
                                            #sim_mat[i][j] = max(0, encoder[i][j] - (e_min + 0.3 * (e_max - e_min))) # multi-news
                                forward_scores = [0 for i in range(len(sim_mat))]
                                backward_scores = [0 for i in range(len(sim_mat))]
                                for i in range(len(sim_mat)):
                                    for j in range(i+1, len(sim_mat[i])):
                                        edge_score = sim_mat[i][j]
                                        forward_scores[j] += edge_score
                                        backward_scores[i] += edge_score
                                scores = []
                                for i in range(size):
                                    scores.append([i, -1.0 * forward_scores[i] + 1.0 * backward_scores[i]]) # cnn/dm
                                    #scores.append([i, -0.7 * forward_scores[i] + 1.7 * backward_scores[i]]) # multi-news

                                sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)     
                                for l in range(size):
                                    selected_ids[k][l] = sorted_scores[l][0]
                                #"""

                                # DASG
                                #"""
                                encoder = np.zeros([size, size])
                                e_max = float("-inf")
                                e_min = float("inf")
                                for i in range(size):
                                    for j in range(size):
                                        if i != j:
                                            # dot, used in cnn/dm tasks
                                            encoder[i][j] = torch.dot(batch_vec[i], batch_vec[j])
                                            # cosine, used in multi-news tasks
                                            #encoder[i][j] = torch.cosine_similarity(batch_vec[i].reshape(1,-1), batch_vec[j].reshape(1,-1))
                                            # manhattan distance
                                            #encoder[i][j] = 100 - torch.dist(batch_vec[i], batch_vec[j], p=1).item()
                                            e_max = max(encoder[i][j], e_max)
                                            e_min = min(encoder[i][j], e_min)
                                for i in range(size):
                                    for j in range(size):
                                        if i != j:
                                            sim_mat[i][j] = max(0, encoder[i][j] - (e_min + 0.05 * (e_max - e_min))) # cnn/dm
                                            #sim_mat[i][j] = max(0, encoder[i][j] - (e_min + 0.8 * (e_max - e_min))) # multi-news

                                m = len(sim_mat) // 3
                                lambdas = [[-1.5, -0.5, -1], [1, 1.5, 2]] # [pos, neg]

                                forward_scores = [0 for i in range(len(sim_mat))]
                                backward_scores = [0 for i in range(len(sim_mat))]
                                for i in range(len(sim_mat)):
                                    for j in range(i+1, len(sim_mat[i])):
                                        # edge_score = sim_mat[i][j]
                                        group_num = min((j - i) // m, 2)
                                        forward_scores[j] += lambdas[0][group_num] * sim_mat[i][j]  # edge_score
                                        backward_scores[i] += lambdas[1][group_num] * sim_mat[j][i]  # edge_score
                                scores = []

                                for i in range(size):
                                    scores.append([i, forward_scores[i] + backward_scores[i]])

                                sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)     
                                for l in range(size):
                                    selected_ids[k][l] = sorted_scores[l][0]
                                #"""

                                # FAR
                                """
                                encoder = np.zeros([size, size])
                                e_max = float("-inf")
                                e_min = float("inf")
                                for i in range(size):
                                    for j in range(size):
                                        if i != j:
                                            # dot
                                            encoder[i][j] = torch.dot(batch_vec[i], batch_vec[j])
                                            # cosine
                                            #encoder[i][j] = torch.cosine_similarity(batch_vec[i].reshape(1,-1), batch_vec[j].reshape(1,-1))
                                            e_max = max(encoder[i][j], e_max)
                                            e_min = min(encoder[i][j], e_min)
                                for i in range(size):
                                    for j in range(size):
                                        if i != j:
                                            sim_mat[i][j] = max(0, encoder[i][j] - (0.1 * (e_max - e_min))) # cnn/dm
                                            #sim_mat[i][j] = max(0, encoder[i][j] - (0.3 * (e_max - e_min))) # multi-news

                                forward_scores = [0 for i in range(len(sim_mat))]
                                backward_scores = [0 for i in range(len(sim_mat))]
                                for i in range(len(sim_mat)):
                                    for j in range(i+1, len(sim_mat[i])):
                                        edge_score = sim_mat[i][j]
                                        forward_scores[j] += edge_score
                                        backward_scores[i] += edge_score
                                scores = []

                                d = F.max_pool1d(batch_vec.T.unsqueeze(1), size, 1).squeeze(-1).squeeze(-1)

                                for i in range(size):
                                    sim_d = torch.cosine_similarity(d.reshape(1, -1), batch_vec[i].reshape(1, -1))
                                    scores.append([i, (sim_d + 1) * (-0.5 * forward_scores[i] + 0.9 * backward_scores[i])]) # cnn/dm
                                    #scores.append([i, (sim_d + 1) * (-0.5 * forward_scores[i] + 2.0 * backward_scores[i])]) # multi-news

                                sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
                                for l in range(size):
                                    selected_ids[k][l] = sorted_scores[l][0]
                                #"""
                                
                        for i, idx in enumerate(selected_ids):
                            _pred = []
                            if (len(batch["src_str"][i]) == 0):
                                continue
                            for j in idx[:len(batch["src_str"][i])]:
                                if (j >= len(batch["src_str"][i])):
                                    continue
                                j = int(j)
                                candidate = batch["src_str"][i][j].strip()
                                # do block_trigram in cnn/dm
                                #"""
                                if (self.args.block_trigram):
                                    if (not _block_tri(candidate, _pred)):
                                        _pred.append(candidate)
                                else:
                                    _pred.append(candidate)
                                """
                                # no block in multi-news
                                _pred.append(candidate)
                                #"""

                                if ((not cal_oracle) and len(_pred) == self.args.select_num):
                                    break
                                
                            # check sentence length only in cnn/dm
                            #"""
                            if (len(batch["tgt_str"][i].replace("<q>", " ").split()) < 30):
                                continue
                            #"""
                            
                            _pred = '<q>'.join(_pred)
                            #_pred = ' '.join(_pred)
                            #_pred = ' '.join(_pred.split()[:len(batch["tgt_str"][i].split())])

                            pred.append(_pred)
                            gold.append(batch["tgt_str"][i])
                            #gold.append(' . '.join(batch["tgt_str"][i].split("<q>")))

                        for i in range(len(gold)):
                            save_gold.write(gold[i].strip() + '\n')
                        for i in range(len(pred)):
                            save_pred.write(pred[i].strip() + '\n')
        rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
        print('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
                    

def test(args, encoder, device, dataloader):
    checkpoint = torch.load(args.model_path, map_location=device)
    print("args training: ", checkpoint['args'])
    to_delete = []
    for k in checkpoint['model'].keys():
        if 'gc3' in k:
            to_delete.append(k)
    for k in to_delete: checkpoint['model'].pop(k)
    
    model = AutoEncoder(args, encoder, device).to(device)
    trainer = Trainer(args, model)
    step = int(args.model_path.split('.')[-2].split('_')[-1])
    trainer.test(dataloader, step)
