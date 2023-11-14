#!/bin/bash
#SBATCH -p cpu -x ironhide

#source activate tf115
source activate py36

python get_importance_index.py -input_path ../nyt/nyt_graph_result -save_path ../nyt/nyt_importance_result -block_size  5 -file_id 3 -test_file_num 0
#python get_importance_index.py -input_path ../nyt_graph_result -save_path ../nyt_importance_result -block_size 5 -file_id 3 -test_file_num 0


# for i in $(seq 0 30)
# do
#    srun -p cpu -J edgecen$i /home/LAB/miniconda3/envs/py36/bin/python -u get_importance_index.py  -file_id=$i \
#        -block_size=5 -input_path /home/LAB/projects/AutoEncoder/preprocess/cnn_graph/ -save_path ../mid_result & sleep 4s
# done

# python get_importance_index.py -input_path /home/LAB/projects/AutoEncoder/preprocess/cnn_graph -save_path /home/LAB/mid_results -block_size  10 -file_id 0 -test_file_num 0