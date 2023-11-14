import torch
import os
import argparse
import torch_geometric as pyg

from dataloader import GAEDataset, gae_collate_fn
from torch.utils.data import DataLoader
from encoder import GCN, GCNII, BiEncoder
from trainer import test
from gcnconv import GCNConvKeyNode, GCN2ConvKeyNode

def getEncoder(args, device):
    embed = torch.nn.Embedding(args.vocab_size, args.embed_size, padding_idx=0) if args.common_embed else None
    if args.encoder == 'gcn':
        if args.conv == 'allnode':
            conv_class = pyg.nn.conv.gcn_conv.GCNConv
            encoder = GCN(args, device, embed=embed, load_embedding=(args.mode=='train'), conv_class=conv_class)
        elif args.conv == 'keynode':
            conv_class = GCNConvKeyNode
            encoder = GCN(args, device, embed=embed, load_embedding=(args.mode=='train'), conv_class=conv_class)
        else:
            encoder1 = GCN(args, device, embed=embed, load_embedding=(args.mode=='train'), conv_class=pyg.nn.conv.gcn_conv.GCNConv)
            encoder2 = GCN(args, device, embed=embed, load_embedding=(args.mode=='train'), conv_class=GCNConvKeyNode)
            if args.train_before_concat:
                return encoder1, encoder2
            encoder = BiEncoder(args, encoder1, encoder2)
    elif args.encoder == 'gcn2':
        if args.conv == 'allnode':
            conv_class = pyg.nn.conv.gcn2_conv.GCN2Conv
            encoder = GCNII(args, device, 16, 0.5, 0.1, 0.5, embed=embed, load_embedding=(args.mode=='train'), conv_class=conv_class)
        elif args.conv == 'keynode':
            conv_class = GCN2ConvKeyNode
            # params_dict = {"nlayers": 16, "dropout":0.5, "alpha":0.1, "theta":0.5}
            encoder = GCNII(args, device, 16, 0.5, 0.1, 0.5, embed=embed, load_embedding=(args.mode=='train'), conv_class=conv_class)
        else:
            encoder1 = GCNII(args, device, 16, 0.5, 0.1, 0.5, embed=embed, load_embedding=(args.mode=='train'), 
                            conv_class=pyg.nn.conv.gcn2_conv.GCN2Conv)
            encoder2 = GCNII(args, device, 16, 0.5, 0.1, 0.5, embed=embed, load_embedding=(args.mode=='train'), conv_class=GCN2ConvKeyNode)
            if args.train_before_concat:
                return encoder1, encoder2
            encoder = BiEncoder(args, encoder1, encoder2)
    else:
        raise Exception("jknet is not implement")
    
    return encoder, embed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-auto_encoder", default='gae', type=str, choices=['gae', 'vgae'])
    parser.add_argument("-pre_loss", default='ce', type=str, choices=['ce', '2ce', 'mse', 'summ_ce'])
    parser.add_argument("-conv", default='allnode', type=str, choices=['allnode', 'keynode', 'both'])
    parser.add_argument("-encoder", default='gcn', type=str, choices=['gcn2', 'jknet'])
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'valid', 'test', 'test_ae'])
    parser.add_argument("-train_before_concat", type=bool ,default=False)
    parser.add_argument("-train_1stage", type=bool ,default=False)

    parser.add_argument("-data_path", default='../../projects/AutoEncoder/presumm_red_data/out_dir_disc_trunc')
    parser.add_argument("-model_path", default='../models/')
    parser.add_argument("-result_path", default='../results/')
    parser.add_argument("-embedding_path", default='../../projects/AutoEncoder/gignore/glove/glove.840B.300d.txt')
    parser.add_argument("-vocab_path", default='../../projects/AutoEncoder/gignore/vocab.txt')
    parser.add_argument("-temp_dir", default='../../projects/AutoEncoder/temp')

    parser.add_argument("-vocab_size", default=50000, type=int)
    parser.add_argument("-embed_size", default=300, type=int)
    parser.add_argument("-hidden_size", default=150, type=int)
    parser.add_argument("-batch_size", default=5, type=int)
    parser.add_argument("-max_sent_len", default=50, type=int)
    
    # parser.add_argument("-use_interval", type=bool, default=True)
    # parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument("-max_tgt_len", default=140, type=int)
    
    parser.add_argument("-dropout", type=float, default=0.1)
    parser.add_argument("-optim_method", default='adam', type=str)
    parser.add_argument("-learning_rate", default=0.1, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)
    parser.add_argument("-loss_lambda", default=0.5, type=float)
    parser.add_argument("-save_checkpoint_steps", default=1000, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-train_epochs", default=2, type=int)
    parser.add_argument("-gae_train_epochs", default=2, type=int)

    parser.add_argument('-num_gpus', default=0, type=int)
    parser.add_argument('-local_rank', default=-1, type=int)
    parser.add_argument('-seed', default=666, type=int)
    parser.add_argument("-block_trigram", type=bool, default=True)
    parser.add_argument("-select_num", type=int, default=3)
    # parser.add_argument("-importance_size", type=int, default=10)#8)

    parser.add_argument("-nlayer_cls", type=int, default=1)
    parser.add_argument("-klloss", type=bool, default=False)
    parser.add_argument("-fgm", type=bool, default=False)
    parser.add_argument("-rdrop", type=bool, default=False)
    parser.add_argument("-rdrop_loss_alpha", type=float, default=0.001)
    parser.add_argument("-encoder_freeze", type=bool, default=False)
    
    parser.add_argument("-alum", type=bool, default=False)
    parser.add_argument("-common_embed", type=bool, default=False)
    parser.add_argument("-noise_var", type=float, default=1e-5)
    parser.add_argument("-project_norm_type", type=str, default="inf", help="l1, l2 or inf")
    parser.add_argument("-noise_gamma", type=float, default=1e-6)
    parser.add_argument("-adv_step_size", type=float, default=1e-3)
    parser.add_argument("-adv_alpha", type=float, default=1)

    parser.add_argument("-arga", type=bool, default=False)
    
    args = parser.parse_args()
    # args.block_trigram = False
    if args.rdrop:
        args.dropout = 0.4

    print("bash args: ", args)
    
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.manual_seed(args.seed)
        device_name = "cuda" 
        device = torch.device(device_name, args.local_rank)
    else:
        device_name = "cpu"
        device = torch.device(device_name)

    print("[INFO] device: ", device)
    gae_data = GAEDataset(args, device)
    loader = DataLoader(gae_data, batch_size=args.batch_size, collate_fn=gae_collate_fn)
    print("[INFO] encoder.py line61 len(loader):", len(loader))

    encoder, embed = getEncoder(args, device)

    if args.mode == "test":
        test(args, encoder, device, loader)
    else:
        raise Exception("only test is available")
    
if __name__ == "__main__":
    main()
