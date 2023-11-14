#!/bin/bash
#SBATCH -p sugon --gres=gpu:1  -o ../slurms/test.log

source activate tf115 
#task=vgae_biencoder_mlp2
#task=vgae_train_before_concat
#task=vgae_mse
task=vgae_mse_concated_neg

echo $task
#model_path=../models/month5_week1/$task/model_summ_step_2000.pt
#model_path=../../projects/AutoEncoder/models/vgae_mse_concated/model_auto_encoder_step_60000.pt

#models in 2022/07-08
#model_path=../../cnn_new_models/vgae_mse_concated_arga-2.5/model_auto_encoder_step_60000.pt
#model_path=../../nyt-0/nyt_models_new50/vgae_mse_concated_arga-nyt2.5-b8e3/model_auto_encoder_step_10000.pt
#model_path=../../multinews_model/vgae_mse_concated_arga-mul2.5-b8e3/model_auto_encoder_step_16000.pt

#models in 2023/04-05
model_path=../../cnn_new_models/vgae_mse_concated_arga-cnn5.5-b8/model_auto_encoder_step_64000.pt
#model_path=../../multinews_model/vgae_mse_concated_arga-mul2.5-b8e3-0418/model_auto_encoder_step_12000.pt

#model_path=../models/$task/model_auto_encoder_step_0.pt
echo $model_path
#cnn/dm
python -u ../src_gae/run.py -batch_size 3 -data_path ../../projects/AutoEncoder/preprocess/cnn_graph/cnndaily -mode test \
    -embedding_path ../../projects/AutoEncoder/gignore/glove/glove.840B.300d.txt -num_gpus 0 -local_rank 0 \
    -vocab_path ../../projects/AutoEncoder/gignore/vocab.txt   -result_path ../results/$task \
    -auto_encoder vgae -model_path $model_path  -conv both -nlayer_cls 2 -select_num 3

#nyt
#python -u ../src_gae/run.py -batch_size 3 -data_path ../../nyt/nyt_importance_result/nyt -mode test \
#    -embedding_path ../../projects/AutoEncoder/gignore/glove/glove.840B.300d.txt -num_gpus 0 -local_rank 0 \
#    -vocab_path ../../projects/AutoEncoder/gignore/vocab.txt   -result_path ../results/$task \
#    -auto_encoder vgae -model_path $model_path  -conv both -nlayer_cls 2 -select_num 3

#multi-news
#python -u ../src_gae/run.py -batch_size 3 -data_path ../../multinews_importance_result/multinews -mode test \
#    -embedding_path ../../projects/AutoEncoder/gignore/glove/glove.840B.300d.txt -num_gpus 0 -local_rank 0 \
#    -vocab_path ../../projects/AutoEncoder/gignore/vocab.txt   -result_path ../results/$task \
#    -auto_encoder vgae -model_path $model_path  -conv both -nlayer_cls 2 -select_num 9
#-encoder gcn2 
