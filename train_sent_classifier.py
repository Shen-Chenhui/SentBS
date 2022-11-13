import os
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

from datasets import load_dataset, load_metric
import datasets
import random
import pandas as pd
import numpy as np
from transformers import AutoTokenizer,AutoModelForSequenceClassification, TrainingArguments, Trainer
import argparse
from tqdm import tqdm
import os

def parse_arguments(parser):
    parser.add_argument('--model_path', type=str, default="", help="model name or path")
    parser.add_argument('--seed', type=int, default=42, help="seed for shuffle")
    parser.add_argument('--num_proc', type=int, default=4, help="number of data processing to run in parallel")
    parser.add_argument('--data_dir', type=str, default="data/classifier_data", help="data directory")
    parser.add_argument('--output_dir', type=str, default="results/classifier", help="output root directory, will be modified to add subdirectories according to model path")

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    # for mred
    idx2labels=["abstract", "strength", "weakness", "suggestion", "ac_disagreement", "rebuttal_process","rating_summary", "decision", "o"]
    labels2idx={
        "abstract":0,
        "strength":1, 
        "weakness":2, 
        "suggestion":3, 
        "ac_disagreement":4, 
        "rebuttal_process":5,
        "rating_summary":6, 
        "decision":7, 
        "O":8
    }

    num_labels=len(labels2idx.keys())

    model_path = args.model_path

    data_dir = args.data_dir
    output_dir = os.path.join(args.output_dir, args.model_path.strip('/').split('/')[-1])

    dataset = load_dataset('csv', data_files=os.path.join(data_dir, 'train.csv'))
    column_names = dataset["train"].column_names
    text_column = column_names[0]
    label_column = column_names[1]
    sorted_dataset = dataset.sort(label_column)
    dataset = sorted_dataset.shuffle(seed=args.seed) # dataset in random order

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    def preprocess_function(examples):
        labels = examples[label_column]
        model_inputs = tokenizer(examples[text_column], truncation=True, max_length=512)
        model_inputs["labels"] = [labels2idx[x] for x in labels]

        return model_inputs


    split_dataset = dataset["train"].train_test_split(test_size=0.1) # NOTE: we only have one file, which is the train here
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"] # NOTE we are not using this for test so its ok
    print("total datasize:", len(dataset["train"]),"; train size:", len(train_dataset),"; val size:", len(val_dataset))

    encoded_train_dataset = train_dataset.map(
        preprocess_function, 
        batched=True,
        remove_columns=column_names,
    )

    encoded_val_dataset = val_dataset.map(
        preprocess_function, 
        batched=True,
        remove_columns=column_names,
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels).to(device)
    metric=load_metric("./accuracy")
    args = TrainingArguments(
        output_dir, # the output directory
        evaluation_strategy = "steps",
        save_strategy = "steps",
        eval_steps = 5000,
        save_steps = 5000,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", # "accuracy"
        greater_is_better=False, # True
        save_total_limit=5,
        overwrite_output_dir=True,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_val_dataset, 
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model()
    # trainer.evaluate()

    # # test
    # model.eval()
    # correct_pred = 0
    # total_pred = 0
    # with open(output_dir+"/result.txt","w", encoding='utf-8') as fw:
    #     fw.write("label\tpred\ttext\n")
    #     for i, data in tqdm(enumerate(encoded_dataset['test'])):
    #         logits = model(input_ids=torch.LongTensor([data['input_ids']]).to(device), attention_mask=torch.LongTensor([data['attention_mask']]).to(device)).logits
    #         pred = torch.argmax(logits, dim=1).item()
    #         if pred == data['label']:
    #             correct_pred += 1
    #         total_pred += 1
    #         fw.write(idx2labels[data['label']]+"\t"+idx2labels[pred]+"\t"+data['text']+'\n')

    # print("accuracy:", correct_pred/total_pred*100)