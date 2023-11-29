
# BiGAE
This is the repository of the paper ***[Bipartite Graph Pre-training for Unsupervised Extractive Summarization with Graph Convolutional Auto-Encoders]([https://arxiv.org/abs/2305.12074](https://arxiv.org/abs/2310.18992))***.

BiGAE, a novel graph pre-training auto-encoder, explicitly models intra-sentential distinctive features and inter-sentential cohesive features through sentence-word bipartite graphs, achieving superior performance in unsupervised summarization frameworks by generating summary-worthy sentence representations that outperform heavy BERT- or RoBERTa-based embeddings in downstream tasks.

## Environment

Run command below to install all the environment in need(**using python3**)

```shell
pip install -r requirements.txt
```

The pyrouge package requires additional installation procedures. If you need to run the extractive summarization task, please refer [this site](https://stackoverflow.com/questions/45894212/installing-pyrouge-gets-error-in-ubuntu) to install pyrouge.

## Data

We provide all datasets used in our experiments:

- The datasets used for upstream and downstream tasks are [**CNN/DailyMail** and **Multi-News**](https://drive.google.com/file/d/1skFKn8GQKWbh7JL3dnuavcP69Vp5tk4z/view?usp=sharing). Please unzip the downloaded file and replace the empty ```./data``` folder.

## Usage

### Upstream Task (Pre-training):

Generate the pre-trained model by executing the following shell script. (Before running, change the "data_path" augument in the script as needed)

```shell
bash pretraining/scripts/run.sh
```

After running finished, execute the following script to test generated pre-trained model.

```shell
bash pretraining/scripts/run.sh
```

-----

### Downstream Task (Summarization):

Using the pre-trained model generated in upstream task, we conduct text summarization. We carry out the tasks by running the following shell script.

```shell
bash summarization/script/test.sh
```

During the experiments, we employed 4 different text summarization methods (we set DASG as default). These methods could be changed in `summarization/src_gae/trainer.py`. For tasks using different datasets, the "data_path" and "model_path" arguments in `test.sh` need to be adjusted.

All the example scripts can be found in `./script`

## Citation

```
@article{jiang2023disco,
  title={DisCo: Distilled Student Models Co-training for Semi-supervised Text Mining},
  author={Jiang, Weifeng and Mao, Qianren and Li, Jianxin and Lin, Chenghua and Yang, Weiyi and Deng, Ting and Wang, Zheng},
  journal={arXiv preprint arXiv:2305.12074},
  year={2023}
}
```
