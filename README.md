# Boosting Data Analytics with Synthetic Volume Expansion
[Arxiv](https://arxiv.org/pdf/2310.17848.pdf) | [BibTex](#bibtex)

Repo containing all raw codes for reproducing results from "Boosting Data Analytics with Synthetic Volume Expansion" by X.Shen, Y. Liu and R. Shen.





This repo is undergoing structural changes for better readability, but most relevant codes for reproducing results can be accessed from:
- [sentiment](https://github.com/yifei-liu-stat/syn/tree/main/sentiment): Sentiment analysis with Syn-Slm.
- [conditional](https://github.com/yifei-liu-stat/syn/tree/main/conditional): Tabular data regression with Syn-Slm.
- [tab-ddpm/synpred](https://github.com/yifei-liu-stat/syn/tree/main/tab-ddpm/synpred): Syn-Boost for predictions on benchmark datasets and simulations.
- [tab-ddpm/syninf](https://github.com/yifei-liu-stat/syn/tree/main/tab-ddpm/syninf): Syn-Test for inference on real datasets and simulations.


The training codes for the tabular diffusion model are mainly adapted from "TabDDPM: Modelling Tabular Data with Diffusion Models" ([paper](https://arxiv.org/abs/2209.15421), [code](https://github.com/yandex-research/tab-ddpm))


## To-Do List
- [ ] Polish the Syn-Boost transfer learning example on Adult dataset
- [ ] Polish the Syn-Test examples
- [ ] Add Makefile for reproducing the results with a pipeline for each example


## Set Up Environment

(Tested on Ubuntu 18.04, with 4 TITAN RTX GPUs and CUDA Version 10.2.)

Create conda environment and install dependencies using the following commands:
```bash
# export REPO_DIR=your/path/to/the/cloned/repo/syn
export REPO_DIR=~/Documents/syn/

conda env create -f environment.yml
conda activate syn
poetry install --no-root
```


## Syn-Slm: Exploring Predictive Powers of Conditional Generative Models


### Sentiment Analysis

Both training and testing results for fine-tuning GPT-3.5 can be found in [gpt_result.csv](https://github.com/yifei-liu-stat/syn/blob/main/sentiment/result/gpt_result.csv). 
Note that we are using GPT-3.5 as a completion model, which is essentially a conditional generaotr, and is in alignment with the central idea of Syn-Slm.

For other approaches, the results can be obtained via:

```bash
cd sentiment/

# download the data
gdown 'https://drive.google.com/file/d/14ixyrPbne9IfD_NCaSkXd7rarasSMgFY/view?usp=drive_link' --fuzzy -O ./data/
gdown 'https://drive.google.com/file/d/15L-hkzSNBVMnC665YXnjO52DPWUS_izi/view?usp=drive_link' --fuzzy -O ./data/

# download the DistilBERT checkpoint
gdown 'https://drive.google.com/file/d/1G8MF5l4LxgOtfXCWiiXrczekfTVlGiMC/view?usp=drive_link' --fuzzy -O ./ckpt/
# # or train on four GPUs and save the checkpoint
# python imdb_distilbert.py --train

# load the saved checkpoint directly for evaluation DistilBERT
python imdb_distilbert.py --predict

# train and evaluate the performance of LSTM
python imdb_lstm.py
```

### Regression Simulation with Tabular Data

Compare with traditional approach (CatBoost) with different $\sigma$ in the underlying model.
```bash
cd conditional/

python ablation_sigma.py --sigma 0.1 --device "cuda:0"
```

## Syn-Boost: Tune Synthetic Size to Achieve Optimal Performance

```bash
cd tab-ddpm/

Download data.zip for data and exp.zip for configurations and checkpoints
```

### Real Datasets -- Identical Distribution
Notebook [prediction_pool_main.ipynb](https://github.com/yifei-liu-stat/syn/blob/main/tab-ddpm/synpred/prediction_pool_main.ipynb) aggregates the results from Syn-Boost on eight benchmark tabular dataset.

For this experiment, [optuna](https://github.com/optuna/optuna) is used to tune the Cat-Boost model trained on synthetic data. To get the result for each of the eight datasets, run (for example):
```bash
python synpred/prediction_pool_tune.py \
    --dsname insurance \
    --maxrho 20 \
    --nratios 10 \
    --ntrials 10 \
    --device "cuda:0" \
```
A corresponding `optuna` study will be saved under the directory `ratio_optuna_studies` containing the tuning results of SynBoost.

### Real Datasets -- Different Distribution
*(WIP: set up REPO_DIR from jupyter notebook)*

- Check notebook **[synpred/transfer_adult/transfer.ipynb]** for pre-trained model on Adult-Male, fine-tuned model on Adult-Female and model performance with various visualizations and distributional distances.
- Notebook **[synpred/transfer_adult/synboost_transfer.ipynb]** implements Syn-Boost method using the fine-tuned tabular diffusion model.



### Simulation Study

Notebook [evaluate.ipynb](https://github.com/yifei-liu-stat/syn/blob/main/tab-ddpm/synpred/sim_prediction/evaluate.ipynb) evaluates the performance of pre-trained generator and investifate the effect of generational error on the performance of Syn-Boost.
To reproduce the results,

```bash
cd synpred/sim_prediction/

# Specify pre-training size, pre-train the model and fine-tune it on the raw training data
python train.py

# Syn-Boost tuning versus CatBoost
python synboost.py

# 2-Wasserstein distance of the pre-trained/fine-tuned generators
python w2_multiprocessing.py
```

## Syn-Test: Tune Synthetic Size to Boost Powers (WIP)



## BibTex

```
@article{shen2023boosting,
  title={Boosting data analytics with synthetic volume expansion},
  author={Shen, Xiaotong and Liu, Yifei and Shen, Rex},
  journal={arXiv preprint arXiv:2310.17848},
  year={2023}
}
```