import os
import re
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from gae import AutoEncoder, Summarizer, MultiTaskSummarizer
from encoder import BiEncoder
from utils import test_rouge, rouge_results_to_str
from aug import Discriminator, alum_loss, FGM
import collections


def build_optimizer(args, model):
    params = model.parameters()
    if args.optim_method == 'sgd':
        optimizer = optim.SGD(params, lr=args.learning_rate)
    elif args.optim_method == 'adagrad':
        optimizer = optim.Adagrad(params, lr=args.learning_rate)
        adagrad_accum = 0
        for group in optimizer.param_groups:
            for p in group['params']:
                optimizer.state[p]['sum'] = optimizer\
                    .state[p]['sum'].fill_(adagrad_accum)
    elif args.optim_method == 'adadelta':
        optimizer = optim.Adadelta(params, lr=args.learning_rate)
    elif args.optim_method == 'adam':
        optimizer = optim.Adam(params, lr=args.learning_rate,
                                    betas=(args.beta1, args.beta2), eps=1e-9)
    else:
        raise RuntimeError("Invalid optim optim_method: " + args.optim_method)
    return optimizer

def build_arga_optimizer(args, model, discriminator):
    params_gen = model.parameters()
    optim_gen = optim.Adam(params_gen, lr=args.learning_rate,
                                    betas=(args.beta1, args.beta2), eps=1e-9)
    params_disc = discriminator.parameters()
    optim_disc = optim.Adam(params_disc, lr=args.learning_rate,
                                    betas=(args.beta1, args.beta2), eps=1e-9)
    return optim_gen, optim_disc
    

def save_model(model, args, optim, model_path, step, model_name=""):
    model_state_dict = model.state_dict()
    checkpoint = {
        'model': model_state_dict,
        'args': args,
        'optim': optim,
    }
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    torch.save(checkpoint, os.path.join(model_path, "model%s_step_%d.pt"%(model_name, step)) )


class Trainer():
    def __init__(self, args, model, embed=None):
        self.args = args
        self.model = model
        self.grad_accum_count = args.accum_count
        self.optim = build_optimizer(args, model)
        self.fgm = FGM(model)
        self.klloss = torch.nn.KLDivLoss(reduction='none')

        self.embed = embed # for alum
        self.discriminator = Discriminator(args)
        self.optim_gen, self.optim_disc = build_arga_optimizer(args, model, self.discriminator)

    def train(self, dataloader, num_epochs, model_name, save=True, output_loss=False):
        """
        The main training loops.
        Args:
            dataloader: data iterator
            num_epochs(int):
            model_name: name for saving checkpoint
            save(bool): save option
            output_loss: whether to print loss
        Return:
            None
        """
        
        accum = 0
        step = 0
        for epoch in range(num_epochs):
            print("train epoch", epoch)
            for batch in dataloader:
                step += 1
                accum += 1
                if accum == 1:
                    self.optim.zero_grad()

                ret = self.model(**batch)
                loss1 = ret["loss"]
                #loss = (loss * mask.float()).sum()
                if self.args.alum and "summ" in model_name:
                    loss1 = alum_loss(self.args, loss1, ret["embed"], self.model, batch, ret["sent_scores"])
                (loss1 / loss1.numel()).backward(retain_graph=True)
                # loss.div(float(normalization)).backward()

                if self.args.rdrop:
                    ret2 = self.model(**batch)
                    loss2 = ret2["loss"]
                    (loss2 / loss2.numel()).backward(retain_graph=True)
                    iden_loss = self.klloss(F.log_softmax(ret["sent_scores"],dim=-1), \
                            F.softmax(ret2["sent_scores"],dim=-1)).sum() + \
                            self.klloss(F.log_softmax(ret2["sent_scores"],dim=-1), \
                            F.softmax(ret["sent_scores"],dim=-1)).sum()
                    iden_loss = iden_loss / 2 * self.args.rdrop_loss_alpha
                
                """
                if self.args.fgm:
                    self.fgm.attack()
                    ret = self.model(**batch)
                    loss2 = ret["loss"]
                    (loss2 / loss2.numel()).backward()
                    self.fgm.restore()
                """

                if self.grad_accum_count == accum:
                    self.optim.step()
                    accum = 0

                if self.args.arga and "auto" in model_name:
                    self.train_arga_step(batch)

                if output_loss: print("loss: ", loss1)
                if save and step % self.args.save_checkpoint_steps == 0:
                    print("[INFO] loss", loss1)
                    print("train step", step)
                    save_model(self.model, self.args, self.optim, self.args.model_path, step, model_name=model_name)
    
            # in case of multi step gradient accumulation,
            # update only after accum batches
            if self.grad_accum_count > 1:
                self.optim.step()
            # save checkpoint
    
    def train_arga_step(self, batch):
        self.optim_gen.zero_grad()
        encoded_emb, _ = self.model.auto_encoder.encode(batch["src"], batch["src_for_sent"],\
                        batch["graph_data"].edge_index, None)
        gen_loss = self.discriminator.gen_loss(encoded_emb)
        gen_loss.backward()
        self.optim_gen.step()

        self.optim_disc.zero_grad()
        encoded_emb, _ = self.model.auto_encoder.encode(batch["src"], batch["src_for_sent"],\
                        batch["graph_data"].edge_index, None)
        disc_loss = self.discriminator.dc_loss(encoded_emb)
        disc_loss.backward()
        self.optim_disc.step()

    
    def test_ae(self, dataloader):
        step = 0
        # acc_pos = []
        # acc_neg = []
        # acc = []
        pos_diff = []
        neg_diff = []
        diff = []
        # f2 = open('data-b1n.txt', 'w')
        with torch.no_grad():
            for batch in dataloader:
                step += 1
                ret = self.model.forward_test(**batch)

                pos_diff.append(ret["pos_diff"])
                neg_diff.append(ret["neg_diff"])
                diff.extend([ret["pos_diff"], ret["neg_diff"]])

                # #将tensor变量转化为numpy类型
                # x = ret["pos_diff"].cpu().numpy()
                # #将numpy类型转化为list类型
                # x=x.tolist()
                # #将list转化为string类型
                # strNums=[str(x_i) for x_i in x]
                # str1=",".join(strNums)
                # #将str类型数据存入本地文件1.txt中
                # f2.write(str1 + "\n")
                # f2.write("data\n")

                # x = batch["graph_data"].edge_index.cpu().numpy()
                # x=x.tolist()
                # strNums=[str(x_i) for x_i in x]
                # str1=",".join(strNums)
                # f2.write(str1 + "\n")
                # f2.write("graph\n")


                # pos_diff += ret["pos_diff"].item()
                # neg_diff += ret["neg_diff"].item()
                # diff += ret["diff"].item()
        #         acc_pos.append(ret["acc_pos"])
        #         acc_neg.append(ret["acc_neg"])
        #         acc.append(ret["acc"])
        #         print("step%d :"%step, ret)
        # print(ret['pos_diff'])
        # print(ret['pos_diff'].shape)
        # f2.close
        pos_diff = torch.cat(pos_diff, dim=0)
        neg_diff = torch.cat(neg_diff, dim=0)
        diff = torch.cat(diff, dim=0)
        print('num_pos_edge: ', pos_diff.size(0), 'avg_pos_diff: ', pos_diff.mean())
        print('num_neg_edge: ', neg_diff.size(0), 'avg_neg_diff: ', neg_diff.mean())
        print('num_edge: ', diff.size(0), 'avg_diff: ', diff.mean())
        # ld = len(acc_pos)
        # print(sum(acc_pos)/ld, sum(acc_neg)/ld, sum(acc)/ld)

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
                            sent_scores = ret["sent_scores"] #从模型获取句子的分数

                            mask = batch["mask_cls"] #遮住padding的句子
                        
                            sent_scores = sent_scores + mask.float()
                            sent_scores = sent_scores.cpu().data.numpy()
                            selected_ids = np.argsort(-sent_scores, 1) #选择分数最高的句子
                        # TODO: 改变选取句子的方法，使用其它的无监督方法
                        for i, idx in enumerate(selected_ids):
                            _pred = []
                            if (len(batch["src_str"][i]) == 0):
                                continue
                            for j in idx[:len(batch["src_str"][i])]:
                                if (j >= len(batch["src_str"][i])):
                                    continue
                                candidate = batch["src_str"][i][j].strip()
                                if (self.args.block_trigram):
                                    if (not _block_tri(candidate, _pred)):
                                        _pred.append(candidate)
                                else:
                                    _pred.append(candidate)

                                if ((not cal_oracle) and len(_pred) == self.args.select_num):
                                    break

                            _pred = '<q>'.join(_pred)
                            #_pred = ' '.join(_pred.split()[:len(batch["tgt_str"][i].split())])

                            pred.append(_pred)
                            gold.append(batch["tgt_str"][i])

                        for i in range(len(gold)):
                            save_gold.write(gold[i].strip() + '\n')
                        for i in range(len(pred)):
                            save_pred.write(pred[i].strip() + '\n')
        rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
        print('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
                    
    
def train(args, encoder, device, dataloader):
    auto_encoder = AutoEncoder(args, encoder, device).to(device) #自编码器
    print("Training auto-encoder")
    trainer = Trainer(args, auto_encoder) # 训练自编码器，无监督，预测词-句子的TF-IDF
    trainer.train(dataloader, args.gae_train_epochs, "_auto_encoder", save=True)
    save_model(auto_encoder, args, None, args.model_path, 0, '_auto_encoder')
    
    summ = Summarizer(args, encoder, device).to(device) # 摘要
    trainer = Trainer(args, summ) 
    print("Training summarizer")
    trainer.train(dataloader, args.train_epochs, "_summ", save=True) #训练摘要模型，有监督，预测是否是摘要句子
    return summ

def train_before_concat(args, encoders, device, dataloader):
    encoder1, encoder2 = encoders
    auto_encoder = AutoEncoder(args, encoder1, device).to(device)
    print("Training auto-encoder1")
    trainer = Trainer(args, auto_encoder)
    trainer.train(dataloader, args.gae_train_epochs, "_auto_encoder1", save=True)

    auto_encoder = AutoEncoder(args, encoder2, device).to(device)
    print("Training auto-encoder2")
    trainer = Trainer(args, auto_encoder)
    trainer.train(dataloader, args.gae_train_epochs, "_auto_encoder2", save=True)

    encoder = BiEncoder(args, encoder1, encoder2)
    summ = Summarizer(args, encoder, device).to(device)
    trainer = Trainer(args, summ)
    print("Training summarizer")
    trainer.train(dataloader, args.train_epochs, "_summ", save=True)
    return summ

def test(args, encoder, device, dataloader):
    checkpoint = torch.load(args.model_path, map_location=device)
    print("args training: ", checkpoint['args'])
    to_delete = []
    for k in checkpoint['model'].keys():
        if 'gc3' in k:
            to_delete.append(k)
    for k in to_delete: checkpoint['model'].pop(k)
    
    #TODO: 加载AutoEncoder，使用自编码器获取句子向量
    model = Summarizer(args, encoder, device, checkpoint["model"])
    model = model.to(device)
    trainer = Trainer(args, model)
    step = int(args.model_path.split('.')[-2].split('_')[-1])
    trainer.test(dataloader, step)

def test_ae(args, encoder, device, dataloader):
    checkpoint = torch.load(args.model_path, map_location=device)
    print("args training: ", checkpoint['args'])

    if "auto" not in args.model_path:
        to_add = []
        for k, v in checkpoint['model'].items():
            print(k)
            if 'encoder' in k:
                to_add.append((k.split('encoder.')[-1], v))
        encoder.load_state_dict(collections.OrderedDict(to_add))
        model = AutoEncoder(args, encoder, device).to(device)
    else:
        model = AutoEncoder(args, encoder, device).to(device)
        model.load_state_dict(checkpoint['model'])
    
    
    trainer = Trainer(args, model)
    trainer.test_ae(dataloader)
