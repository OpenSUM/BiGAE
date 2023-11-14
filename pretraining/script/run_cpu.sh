#!/bin/bash
#SBATCH -p cpu  -o ../slurms/run_cpu_061402.log
#SBATCH -J arga

source activate tf115
task=small_arga
#task=small_klloss
#keynode_small_gcn2_allnode
echo $task
python -u ../src_gae/run.py -batch_size 8  -mode train -learning_rate 1e-4 \
    -embedding_path ../gignore/glove/glove.840B.300d.txt -data_path ../preprocess/tmp/cnndaily \
    -vocab_path ../gignore/vocab.txt -model_path ../models/$task  -result_path ../results/$task\
    -save_checkpoint_steps 500  -num_gpus 1 -local_rank -1 -train_epochs 1  -gae_train_epochs 1 \
    -auto_encoder vgae -loss_lambda 0.1  -nlayer_cls 2  -pre_loss mse  -conv both -arga true
#-alum true
#-klloss true -train_before_concat true -conv both
#-block_trigram False -pre_loss summ_ce
#-encoder gcn2
