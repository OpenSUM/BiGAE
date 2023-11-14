#!/bin/bash
#SBATCH -p cpu  -o ../slurms/test_cpu_230411.log -x dinobots

source activate tf115 
#task=keynode_small_lr1e-4_2neg
#task=keynode_small_lr1e-4_losslambda
#task=keynode_lr1e-4_mlp2
#task=vgae_gnc2
#task=keynodeR_lr1e-4_noae
#task=vgae_mlp3
#task=vgae_noblocktri
#task=vgae_biencoder_mlp2
#task=vgae_longtrain
#task=vgae_2ce
#task=vgae_mse_concated_1stage_lr5e-4_weighted
#task=vgae_mse_concated_arga
task=vgae_mse_concated_arga-1.5-b32
#nlr1e-5
#rdrop_alpha0001
#nseed13131
#_lr7e-5
#_lr2e-4

echo $task
#model_path=../models/$task/model_summ_step_60000.pt
#model_path=../models/$task/model_summ_step_0.pt
model_path=../cnn_new_models/$task/model_auto_encoder_step_0.pt
echo $model_path
python -u ../src_gae/run.py -batch_size 3 -data_path ../preprocess/cnn_graph/cnndaily -mode test\
    -embedding_path ../gignore/glove/glove.840B.300d.txt -num_gpus 0 -local_rank -1 \
    -vocab_path ../gignore/vocab.txt   -result_path ../results/$task \
    -auto_encoder vgae -model_path $model_path  -conv both -nlayer_cls 2
#-data_path ../preprocess/cnn_graph/cnndaily
#-arga true
#-pre_loss 2ce
#-encoder gcn2 
