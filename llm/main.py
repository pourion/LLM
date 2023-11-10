import pdb

import torch
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import textwrap
import matplotlib.pyplot as plt

train_dataset = load_dataset("tatsu-lab/alpaca", split="train")

print(train_dataset)
# We can get the first five rows as follows
pandas_format = train_dataset.to_pandas()
print(pandas_format.head())
for index in range(3):
    print("---" * 15)
    print(
        "Instruction: {}".format(
            textwrap.fill(pandas_format.iloc[index]["instruction"], width=50)
        )
    )
    print(
        "Output: {}".format(
            textwrap.fill(pandas_format.iloc[index]["output"], width=50)
        )
    )
    print("Text: {}".format(textwrap.fill(pandas_format.iloc[index]["text"], width=50)))


pandas_format["text_length"] = pandas_format["text"].apply(len)

max_length = pandas_format["text_length"].max()

plt.figure(figsize=(10, 6))
plt.hist(pandas_format["text_length"], bins=50, alpha=0.5, color="g")

plt.annotate(
    "Max length: {}".format(max_length),
    xy=(max_length, 0),
    xytext=(max_length, 50),
    arrowprops=dict(facecolor="red", shrink=0.05),
)

plt.title("Distribution of Length of Text")
plt.xlabel("Length of Text")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("test.png")


mask = pandas_format["text_length"] > 1024
percentage = (mask.sum() / pandas_format["text_length"].count()) * 100

print(
    f"The percentage of text documents with a length greater than 1024 is: {percentage}%"
)


# --- Model training
pretrained_model_name = "Salesforce/xgen-7b-8k-base"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name,
    torch_dtype=torch.bfloat16,
    resume_download=True,
)
model_training_args = TrainingArguments(
    output_dir="xgen-7b-8k-base-fine-tuned",
    per_device_train_batch_size=4,
    optim="adamw_torch",
    logging_steps=80,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    num_train_epochs=1,
    save_strategy="epoch",
)

lora_peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM"
)

SFT_trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=model_training_args,
    packing=True,
    peft_config=lora_peft_config,
)

tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))
model = prepare_model_for_int8_training(model)
model = get_peft_model(model, lora_peft_config)

training_args = model_training_args
trainer = SFT_trainer
trainer.train()
