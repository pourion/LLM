import torch
import numpy as np

from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import Trainer
import evaluate

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_datasets = load_dataset("glue", "mrpc")
raw_train_dataset = raw_datasets["train"]



def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
batch = data_collator(samples)


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
# sequences = [
#     "I've been waiting for a HuggingFace course my whole life.",
#     "This course is amazing!",
# ]
# batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
# batch["labels"] = torch.tensor([1, 1])
# optimizer = AdamW(model.parameters())
# loss = model(**batch).loss
# loss.backward()
# optimizer.step()

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments("results/test_bert_finetuning", evaluation_strategy="epoch")
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)



predictions = trainer.predict(tokenized_datasets["validation"])
preds = np.argmax(predictions.predictions, axis=-1)

metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)

trainer.train()