import pdb

import torch
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import textwrap


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
