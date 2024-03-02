# Boosting Data Analytics with Synthetic Volume Expansion
[Arxiv](https://arxiv.org/pdf/2310.17848.pdf) | [BibTex](#bibtex)

Repo containing all raw codes for reproducing results from "Boosting Data Analytics with Synthetic Volume Expansion" by X.Shen, Y. Liu and R. Shen.







This repo is undergoing structural changes for better readability, but most relevant codes for reproducing results can be found in folders [tab-ddpm/syn](https://github.com/yifei-liu-stat/syn/tree/main/tab-ddpm/syn) and [conditional](https://github.com/yifei-liu-stat/syn/tree/main/conditional).


The training codes for the tabular diffusion model are mainly adapted from "TabDDPM: Modelling Tabular Data with Diffusion Models" ([paper](https://arxiv.org/abs/2209.15421), [code](https://github.com/yandex-research/tab-ddpm))

## Experiment (WIP)

(Tested on Ubuntu 18.04, with 4 TITAN RTX GPUs and CUDA Version 10.2.)

Create conda environment and install dependencies using the following commands:
```bash
conda env create -f environment.yml
conda activate syn
poetry install --no-root
```


### Syn-Slm: Exploring Predictive Powers of Conditional Generative Models


#### Sentiment Analysis

Both training and testing results for fine-tuning GPT-3.5 can be found in `sentiment/result/gpt_result.csv`. 
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



## BibTex

```
@article{shen2023boosting,
  title={Boosting data analytics with synthetic volume expansion},
  author={Shen, Xiaotong and Liu, Yifei and Shen, Rex},
  journal={arXiv preprint arXiv:2310.17848},
  year={2023}
}
```