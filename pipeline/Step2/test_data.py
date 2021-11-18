from functools import partial

import numpy as np
import paddle
import pandas as pd
import torch
from datasets import Dataset
from paddlenlp.data import Dict, Pad, Stack
from paddlenlp.datasets import load_dataset as ppnlp_load_dataset
from paddlenlp.transformers import BertTokenizer as PPNLPBertTokenizer
from reprod_log import ReprodDiffHelper, ReprodLogger
from transformers import BertTokenizer as HFBertTokenizer
from transformers import DataCollatorWithPadding


def build_paddle_data_pipeline():
    def read(data_path):
        df = pd.read_csv(data_path, sep="\t")
        for _, row in df.iterrows():
            yield {"sentence": row["sentence"], "labels": row["label"]}

    def convert_example(example, tokenizer, max_length=128):
        labels = np.array([example["labels"]], dtype="int64")
        example = tokenizer(example["sentence"], max_seq_len=max_length)
        return {
            "input_ids": np.array(
                example["input_ids"], dtype="int64"),
            "token_type_ids": np.array(
                example["token_type_ids"], dtype="int64"),
            "labels": labels,
        }

    # load tokenizer
    tokenizer = PPNLPBertTokenizer.from_pretrained("bert-base-uncased")
    # load data
    dataset_test = ppnlp_load_dataset(
        read, data_path='demo_sst2_sentence/demo.tsv', lazy=False)
    trans_func = partial(convert_example, tokenizer=tokenizer, max_length=128)
    # tokenize data
    dataset_test = dataset_test.map(trans_func, lazy=False)
    collate_fn = lambda samples, fn=Dict({
        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        "labels": Stack(dtype="int64"), }): fn(samples)
    test_sampler = paddle.io.SequenceSampler(dataset_test)
    test_batch_sampler = paddle.io.BatchSampler(
        sampler=test_sampler, batch_size=4)
    data_loader_test = paddle.io.DataLoader(
        dataset_test,
        batch_sampler=test_batch_sampler,
        num_workers=0,
        collate_fn=collate_fn, )

    return dataset_test, data_loader_test


def build_torch_data_pipeline():
    tokenizer = HFBertTokenizer.from_pretrained("bert-base-uncased")

    def preprocess_function(examples):
        result = tokenizer(
            examples["sentence"],
            padding=False,
            max_length=128,
            truncation=True,
            return_token_type_ids=True, )
        if "label" in examples:
            result["labels"] = [examples["label"]]
        return result

    # load data
    dataset_test = Dataset.from_csv("demo_sst2_sentence/demo.tsv", sep="\t")
    dataset_test = dataset_test.map(
        preprocess_function,
        batched=False,
        remove_columns=dataset_test.column_names,
        desc="Running tokenizer on dataset", )
    dataset_test.set_format(
        "np", columns=["input_ids", "token_type_ids", "labels"])
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    collate_fn = DataCollatorWithPadding(tokenizer)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=4,
        sampler=test_sampler,
        num_workers=0,
        collate_fn=collate_fn, )
    return dataset_test, data_loader_test


def test_data_pipeline():
    diff_helper = ReprodDiffHelper()
    paddle_dataset, paddle_dataloader = build_paddle_data_pipeline()
    torch_dataset, torch_dataloader = build_torch_data_pipeline()

    logger_paddle_data = ReprodLogger()
    logger_torch_data = ReprodLogger()

    logger_paddle_data.add("length", np.array(len(paddle_dataset)))
    logger_torch_data.add("length", np.array(len(torch_dataset)))

    # random choose 5 images and check
    for idx in range(5):
        rnd_idx = np.random.randint(0, len(paddle_dataset))
        for k in ["input_ids", "token_type_ids", "labels"]:

            logger_paddle_data.add(f"dataset_{idx}_{k}",
                                   paddle_dataset[rnd_idx][k])

            logger_torch_data.add(f"dataset_{idx}_{k}",
                                  torch_dataset[rnd_idx][k])

    for idx, (paddle_batch, torch_batch
              ) in enumerate(zip(paddle_dataloader, torch_dataloader)):
        if idx >= 5:
            break
        for i, k in enumerate(["input_ids", "token_type_ids", "labels"]):
            logger_paddle_data.add(f"dataloader_{idx}_{k}",
                                   paddle_batch[i].numpy())
            logger_torch_data.add(f"dataloader_{idx}_{k}",
                                  torch_batch[k].cpu().numpy())

    diff_helper.compare_info(logger_paddle_data.data, logger_torch_data.data)
    diff_helper.report()


if __name__ == "__main__":
    test_data_pipeline()
