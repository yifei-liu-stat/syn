from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from scipy.special import expit

from tqdm import tqdm
from copy import deepcopy
import pickle
import argparse
import os

from sklearn import metrics
from sklearn.metrics import classification_report


import json
import sys


# Prepare IMDB dataset
class IMDBdataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        target = self.targets[idx]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": torch.tensor(target, dtype=torch.long),
        }


class litDistillBERT(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx, prefix="val")

    def _shared_eval_step(self, batch, batch_idx, prefix):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=targets
        )
        loss = outputs["loss"]
        logits = outputs["logits"]

        preds = logits.argmax(dim=1, keepdim=True)
        acc = preds.eq(targets.view_as(preds)).sum().item() / len(preds)
        # log both training loss and accuracy at the end of each epoch, and show them with progress bar
        self.log(f"{prefix}_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log(f"{prefix}_acc", acc, on_epoch=True, on_step=False, prog_bar=True)

        return loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def imdb_jsonl_to_np(jsonl_data_path):
    reviews, sentiments = [], []
    with open(jsonl_data_path, "r") as file:
        temp = file.__iter__()
        while True:
            try:
                json_entry = json.loads(next(temp))
                review, sentiment = json_entry["prompt"], json_entry["completion"]
                review = review.rstrip("\n\n###\n\n")
                sentiment = (
                    1 if sentiment == " 1" else 0
                )  # 1 for positive, 0 for negative
                reviews.append(review)
                sentiments.append(sentiment)

            except StopIteration:
                print("Read file exhausted.")
                break

    return np.array(reviews), np.array(sentiments)


def prepare_dl(reviews, sentiments, tokenizer):
    ds = IMDBdataset(
        reviews=reviews, targets=sentiments, tokenizer=tokenizer, max_len=512
    )
    dl = DataLoader(
        ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, shuffle=True
    )
    return dl


ckpt_dir = "ckpt/"
ckpt_name = "imdb_distilbert_ckpt"


BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-5
MAX_LEN = 512

train_data_path = "data/imdb_prepared_train.jsonl"  # size about 49,000
test_data_path = "data/imdb_prepared_valid.jsonl"  # size about 1,000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune DistillBERT to perform sentiment classification on IMDB review dataset."
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t", "--train", action="store_true")
    group.add_argument("-i", "--infer", action="store_true")
    group.add_argument("-p", "--predict", action="store_true")

    args = parser.parse_args()

    temp_reviews, temp_sentiments = imdb_jsonl_to_np(train_data_path)

    train_reviews, train_sentiments = (
        temp_reviews[:45000],
        temp_sentiments[:45000],
    )  # 45,000
    val_reviews, val_sentiments = temp_reviews[45000:], temp_sentiments[45000:]  # 3,576
    test_reviews, test_sentiments = imdb_jsonl_to_np(test_data_path)  # 1,000


    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dl_train = prepare_dl(train_reviews, train_sentiments, tokenizer)
    dl_val = prepare_dl(val_reviews, val_sentiments, tokenizer)
    dl_test = prepare_dl(test_reviews, test_sentiments, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2, output_attentions=True
    )
    litmodel = litDistillBERT(model=model, lr=LEARNING_RATE)

    if args.train:
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            mode="min",
            monitor="val_loss",
            dirpath=ckpt_dir,
            filename=ckpt_name,
        )

        trainer = Trainer(
            max_epochs=EPOCHS,
            accelerator="gpu",
            devices=[4, 5, 6, 7],
            strategy="ddp",
            precision="16",
            deterministic=True,
            callbacks=[checkpoint_callback],
            default_root_dir="./",
        )

        trainer.fit(litmodel, dl_train, dl_val)

    if args.infer:
        trainer = Trainer(
            max_epochs=EPOCHS,
            accelerator="gpu",
            devices=[7],
            precision="16",
            deterministic=True,
        )

        checkpoint_name = os.listdir(ckpt_dir)[0]

        trainer.validate(
            model=litmodel,
            dataloaders=dl_test,
            ckpt_path=ckpt_dir + checkpoint_name,
        )

    if args.predict:
        device = "cuda:0"
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2, output_attentions=True
        )

        for ckpt_name in os.listdir(ckpt_dir):
            if ckpt_name.startswith("imdb-distillbert"):
                checkpoint_name = ckpt_name
                print(ckpt_name)
                break

        litmodel = litDistillBERT.load_from_checkpoint(
            os.path.join(ckpt_dir, checkpoint_name),
            model=model,
            lr=LEARNING_RATE,
        ).to(device)
        litmodel.eval()

        pred_dict = {"preds": [], "targets": [], "probs": []}
        for batch in tqdm(dl_test):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            outputs = litmodel.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=targets
            )
            loss = outputs["loss"]
            logits = outputs["logits"]

            preds = logits.argmax(dim=1, keepdim=True)
            preds = preds.view_as(targets)

            pred_dict["preds"].extend(preds.tolist())
            pred_dict["targets"].extend(targets.tolist())
            pred_dict["probs"].extend(expit(logits[:, 1].detach().tolist()))

        pickle.dump(pred_dict, open("result/pred_dict.pkl", "wb"))

        # Performance metrics
        true_list = pred_dict["targets"]
        pred_list = pred_dict["preds"]
        prob_list = pred_dict["probs"]
        print(classification_report(true_list, pred_list, digits=3))
        
        
        fpr, tpr, thresholds = metrics.roc_curve(true_list, prob_list, pos_label=1)
        auroc = metrics.auc(fpr, tpr)
        print("Area under ROC curve:", auroc)

        pr, re, thresholds = metrics.precision_recall_curve(true_list, prob_list, pos_label=1)
        auprc = metrics.auc(re, pr)
        print("Area under PR curve:", auprc)
