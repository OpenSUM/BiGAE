#!/bin/bash
#SBATCH    -o ../slurm/0822_save_step/0822_cnn_auto5.5-4000.log -p cpu 
#SBATCH    -x dell-gpu-09,dell-gpu-18,dell-gpu-24,dell-gpu-04,dell-gpu-01,dell-gpu-07,dell-gpu-11,dell-gpu-13,dell-gpu-22,dell-gpu-05


source activate tf115 
#task=vgae_biencoder_mlp2
#task=vgae_train_before_concat
#task=vgae_mse
#task=vgae_mse_concated_arga-1.5
task=vgae_mse_concated_arga-5.5

echo $task
#model_path=../models/month5_week1/$task/model_summ_step_2000.pt
#model_path=../nyt-0/nyt_models_new50/$task/model_summ_step_10000.pt
model_path=../cnn_new_models/$task/model_auto_encoder_step_4000.pt
echo $model_path

# python -u ../src_gae/run.py -batch_size 3 -data_path ../preprocess/cnn_graph/cnndaily -mode test \
#     -embedding_path ../gignore/glove/glove.840B.300d.txt -num_gpus 0 -local_rank 0 \
#     -vocab_path ../gignore/vocab.txt   -result_path ../results/$task \
#     -auto_encoder vgae -model_path $model_path  -conv both -nlayer_cls 2 
#-encoder gcn2 

#../preprocess/tmp_cnndmpt_data/cnndaily

python -u /home/LAB/src_gae/run.py -batch_size 3 \
    -data_path ../cnn_mid_result_new/cnndaily -mode test_ae \
    -embedding_path  /home/LAB/projects/AutoEncoder/gignore/glove/glove.840B.300d.txt -num_gpus 0 -local_rank -1 \
    -vocab_path /home/LAB/projects/AutoEncoder/gignore/vocab.txt   -result_path ../20220808_result/$task \
    -auto_encoder vgae -model_path $model_path  -conv both -nlayer_cls 2 

# -o ../slurm/0822_save_step/0817_nyt_auto2.5-b8e8.log --gres=gpu:1 -p sugon 
#-o ../slurm/0822_save_step/0818_cnn_auto1.5-b8e3.log -p cpu 
