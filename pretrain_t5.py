import argparse
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings('ignore')


args_dict = dict(
    data_dir='data/conceptnet_df',  # path for data files
    output_dir="",  # path to save the checkpoints
    model_name_or_path='t5-small',
    tokenizer_name_or_path='t5-small',
    max_seq_length=512,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=8,
    eval_batch_size=8,
    num_train_epochs=5,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=True,  # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1',
    max_grad_norm=1,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ConceptnetDataset(Dataset):
    def __init__(self, tokenizer, dataset, max_len=512):
        self.data = dataset
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.tokenizer.max_length = max_len
        self.tokenizer.model_max_length = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        source_text = str(self.data['input'][index])
        target_text = str(self.data['target'][index])

        input_ = source_text.lower() + ' </s>'
        target = target_text.lower() + ' </s>'

        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [input_], max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        # tokenize targets
        tokenized_targets = self.tokenizer.batch_encode_plus(
            [target], max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )

        source_ids = tokenized_inputs["input_ids"].squeeze()
        target_ids = tokenized_targets["input_ids"].squeeze()

        src_mask = tokenized_inputs["attention_mask"].squeeze()  # might need to squeeze
        target_mask = tokenized_targets["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}


class T5Pretrainer(pl.LightningModule):
    def __init__(self, hparam):
        super(T5Pretrainer, self).__init__()
        self.hparam = hparam

        self.model = T5ForConditionalGeneration.from_pretrained(
            hparam.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            hparam.model_name_or_path
        )
        self.save_hyperparameters()

    def is_logger(self):
        return True

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        # tensorboard_logs = {"train_loss": loss}
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        perplexity = torch.exp(avg_train_loss)
        # tensorboard_logs = {"avg_train_loss": avg_train_loss, "train_perplexity": torch.exp(avg_train_loss)}
        self.log("avg_train_loss", avg_train_loss, prog_bar=True, logger=True)
        self.log("train_perplexity", perplexity, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        perplexity = torch.exp(avg_loss)
        # tensorboard_logs = {"val_loss": avg_loss, "val_perplexity": torch.exp(avg_loss)}
        self.log("val_loss", avg_loss, prog_bar=True, logger=True)
        self.log("val_perplexity", perplexity, prog_bar=True, logger=True)

    def configure_optimizers(self):
        # Prepare optimizer and schedule (linear warmup and decay)

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparam.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparam.learning_rate, eps=self.hparam.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self,
                       epoch=None,
                       batch_idx=None,
                       optimizer=None,
                       optimizer_idx=None,
                       optimizer_closure=None,
                       on_tpu=None,
                       using_native_amp=None,
                       using_lbfgs=None
                       ):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(
            self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(
            tokenizer=self.tokenizer, type_path="train", args=self.hparam)
        print("Train set: ", train_dataset.__len__())
        dataloader = DataLoader(
            train_dataset, batch_size=self.hparam.train_batch_size,
            drop_last=True, shuffle=True, num_workers=2)
        t_total = (
                (len(dataloader.dataset) //
                 (self.hparam.train_batch_size * max(1, self.hparam.n_gpu)))
                // self.hparam.gradient_accumulation_steps
                * float(self.hparam.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparam.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(
            tokenizer=self.tokenizer, type_path="validation", args=self.hparam)
        print("Val dataset: ", val_dataset.__len__())
        return DataLoader(val_dataset, batch_size=self.hparam.eval_batch_size, num_workers=2)


def get_dataset(tokenizer, type_path, args):
    print(f"Read {type_path} data")
    tokenizer.max_length = args.max_seq_length
    tokenizer.model_max_length = args.max_seq_length
    if type_path == "train":
        dataset = pd.read_csv(os.path.join(args_dict['data_dir'], "train.csv"))
    else:
        dataset = pd.read_csv(os.path.join(args_dict['data_dir'], "val.csv"))
    return ConceptnetDataset(tokenizer=tokenizer, dataset=dataset)


def generate_data():
    df = pd.read_csv('data/masked_input_target.csv')
    temp_df = df
    train_dataset = temp_df.sample(frac=0.6, random_state=42)
    temp_df = temp_df.drop(train_dataset.index).reset_index(drop=True)
    val_dataset = temp_df.sample(frac=0.5, random_state=42)
    test_dataset = temp_df.drop(val_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    val_dataset = val_dataset.reset_index(drop=True)
    print(train_dataset.shape)
    print(val_dataset.shape)
    print(test_dataset.shape)
    train_dataset.to_csv('data/conceptnet_df/train.csv')
    val_dataset.to_csv('data/conceptnet_df/val.csv')
    test_dataset.to_csv('data/conceptnet_df/test.csv')


if __name__ == "__main__":
    set_seed(42)
    args = argparse.Namespace(**args_dict)

    print("Create model")
    model = T5Pretrainer(args)

    training_start_time = time.time()
    local_start_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(training_start_time))
    print("Start training " + local_start_time_str)

    checkpoint_callback = ModelCheckpoint(
        dirpath="output/conceptnet/checkpoints",
        filename=f"t5_pretrainer_model_{local_start_time_str}",
        monitor="val_loss", mode="min", save_top_k=5
    )
    logger = TensorBoardLogger("output/conceptnet/training_logs",
                               name=f"t5_pretrainer_model_{local_start_time_str}")

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        # early_stop_callback=False,
        precision=16 if args.fp_16 else 32,
        # amp_level=args.opt_level,
        enable_checkpointing=True,
        gradient_clip_val=args.max_grad_norm,
        callbacks=checkpoint_callback,
        logger=logger,
    )

    trainer = pl.Trainer(**train_params)
    trainer.fit(model)
    print("Finished training")
