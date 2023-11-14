#!/bin/bash
#SBATCH -o ../slurm/run_small_0705_1.log 
#SBATCH -p sugon -x dell-gpu-31 --gres=gpu:1 

source activate tf115
task=small_weight_neg
echo $task
# python -u ../src_gae/run.py -batch_size 8  -mode train -learning_rate 1e-4 \
#     -embedding_path ../gignore/glove/glove.840B.300d.txt -data_path ../projects/AutoEncoder/preprocess/tmp_cnndmpt_data/cnndm_bert.pt \
#     -vocab_path ../gignore/vocab.txt -model_path ../models/$task  -result_path ../results/$task \
#     -save_checkpoint_steps 500  -num_gpus 1 -local_rank 1 -train_epochs 5  -gae_train_epochs 5 \
#     -auto_encoder vgae -loss_lambda 0.1  -nlayer_cls 1  -conv both -pre_loss mse
    
python -u /home/LAB/src_gae/run.py -batch_size 8  -mode train -learning_rate 1e-4 \
    -embedding_path /home/LAB/projects/AutoEncoder/gignore/glove/glove.840B.300d.txt \
    -data_path /home/LAB/projects/AutoEncoder/preprocess/tmp_cnndmpt_data/cnndaily \
    -vocab_path /home/LAB/projects/AutoEncoder/gignore/vocab.txt -model_path ../models/$task  \
    -result_path ../results/$task \
    -save_checkpoint_steps 500  -num_gpus 1 -local_rank 0 -train_epochs 5  -gae_train_epochs 5 \
    -auto_encoder vgae -loss_lambda 0.1  -nlayer_cls 1  -conv both -pre_loss mse

#-klloss true -train_before_concat true 
#-block_trigram False -pre_loss summ_ce
#-encoder gcn2
