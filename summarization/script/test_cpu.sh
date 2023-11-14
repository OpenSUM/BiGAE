#!/bin/bash
#SBATCH -p cpu -x dinobots -o ../slurms/test_cpu.log

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
task=vgae_mse_concated_arga
#nlr1e-5
#rdrop_alpha0001
#nseed13131
#_lr7e-5
#_lr2e-4

echo $task
#model_path=../models/small_arga/model_auto_encoder_step_500.pt
#model_path=../../projects/AutoEncoder/models/vgae_mse_concated/model_auto_encoder_step_60000.pt

#models in 2022/07-08
#model_path=../../cnn_new_models/vgae_mse_concated_arga-2.5/model_auto_encoder_step_60000.pt
#model_path=../../nyt-0/nyt_models_new50/vgae_mse_concated_arga-nyt2.5-b8e3/model_auto_encoder_step_10000.pt
model_path=../../multinews_model/vgae_mse_concated_arga-mul2.5-b8e3/model_auto_encoder_step_16000.pt

#model_path=../models/$task/model_summ_step_0.pt
#model_path=../models/$task/model_auto_encoder_step_0.pt
echo $model_path
#cnn/dm
#python -u ../src_gae/run.py -batch_size 3 -data_path ../../projects/AutoEncoder/preprocess/cnn_graph/cnndaily -mode test\
#    -embedding_path ../../projects/AutoEncoder/gignore/glove/glove.840B.300d.txt -num_gpus 0 -local_rank -1 \
#    -vocab_path ../../projects/AutoEncoder/gignore/vocab.txt   -result_path ../results/$task \
#    -auto_encoder vgae -model_path $model_path  -conv both -nlayer_cls 2 -select_num 3

#nyt
#python -u ../src_gae/run.py -batch_size 3 -data_path ../../nyt_importance_result_new50/nyt -mode test \
#    -embedding_path ../../projects/AutoEncoder/gignore/glove/glove.840B.300d.txt -num_gpus 0 -local_rank -1 \
#    -vocab_path ../../projects/AutoEncoder/gignore/vocab.txt   -result_path ../results/$task \
#    -auto_encoder vgae -model_path $model_path  -conv both -nlayer_cls 2 -select_num 3

#multi-news
python -u ../src_gae/run.py -batch_size 3 -data_path ../../multinews_importance_result/multinews -mode test \
    -embedding_path ../../projects/AutoEncoder/gignore/glove/glove.840B.300d.txt -num_gpus 0 -local_rank -1 \
    -vocab_path ../../projects/AutoEncoder/gignore/vocab.txt   -result_path ../results/$task \
    -auto_encoder vgae -model_path $model_path  -conv both -nlayer_cls 2 -select_num 9
#-arga true
#-pre_loss 2ce
#-encoder gcn2 
