#!/bin/bash
#SBATCH -o ../slurm/run_0422-cnn-5.5-b32.log --gres=gpu:1 
#-p sugon

source activate tf115
#task=vgae_2neg
#task=vgae_mlp3
#task=vgae_biencoder_mlp2_klloss
#task=vgae_longtrain
#task=vgae_mse_concated_1stage_lr5e-4_weighted
task=vgae_mse_concated_arga-cnn5.5-b32
#task = nyt_model_1.5
# task=vgae_mse_concated_arga-mul2.5-b8e3-0418-allnode
#_alpha10
#_nlr1e-5
#alum 
echo $task

python -u ../src_gae/run.py -batch_size 32  -mode train -learning_rate 5e-5 \
    -embedding_path /home/LAB/projects/AutoEncoder/gignore/glove/glove.840B.300d.txt \
    -data_path ../cnn_importance_result/cnndaily \
    -vocab_path /home/LAB/projects/AutoEncoder/gignore/vocab.txt -model_path ../cnn_new_models/$task/  \
    -save_checkpoint_steps 2000  -num_gpus 1  -train_epochs 3  -gae_train_epochs 3 \
    -auto_encoder vgae -loss_lambda 0.1 -nlayer_cls 2 -conv both  -local_rank 0 -pre_loss mse
# -data_path ../cnn_mid_result_new/cnndaily \
#-alum true -adv_alpha 10
#-rdrop true
#seed 474 247 908 1002 17171 13131 614
#-klloss true 
#-train_before_concat true 
#-block_trigram false -pre_loss summ_ce 
#-encoder gcn2
#-data_path ../tmp_data/cnndaily
