from glob import glob
import os 
import pandas as pd 
import shutil
from itertools import chain
from tqdm import tqdm
from pathlib import Path

from transformers import Trainer, TrainingArguments
import random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType
from typing import Optional
import torch

from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
target_dir_list = ['qalora_tra.json'
                   ]

all_json_path = [glob(i+"*.json") for i in target_dir_list]
all_json_path = list(chain(*all_json_path))
len(all_json_path), all_json_path[:5]

def read_json(x:str):
    try:
        data = pd.read_json(x)
        return data 
    except Exception as e:
        return pd.DataFrame()

alldata = pd.concat([read_json(i) for i in all_json_path])

genrate_data_dir = "data3_0328"
genrate_data_dir = Path(genrate_data_dir)

if genrate_data_dir.exists():
    shutil.rmtree(genrate_data_dir, ignore_errors=True)

os.makedirs(genrate_data_dir, exist_ok=True)

alldata = alldata.sample(frac=1).reset_index(drop=True)

chunk_size = 666

for index, start_id in tqdm(enumerate(range(0, alldata.shape[0], chunk_size))):
    temp_data = alldata.iloc[start_id:(start_id+chunk_size)]
    temp_data.to_csv(genrate_data_dir.joinpath(f"{index}.csv"), index=False)


tokenizer = AutoTokenizer.from_pretrained("./chatglm6b-dddd", trust_remote_code=True)

model = AutoModel.from_pretrained(
    "./chatglm6b-dddd", trust_remote_code=True).half().cuda()

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
    # ['dense','dense_h_to_4h','dense_4h_to_h'] # 'query_key_value',
    target_modules=['query_key_value',],
)
model = get_peft_model(model, peft_config)


class MyTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        def save_tunable_parameters(model, path):
            saved_params = {
                k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
            }
            # saved_params = model.state_dict()
            torch.save(saved_params, path)

        save_tunable_parameters(
            self.model, os.path.join(output_dir, "chatglm-lora.pt")
        )


random.seed(42)

all_file_list = glob(pathname=genrate_data_dir.joinpath("*.csv").__str__())

test_file_list = random.sample(all_file_list, int(len(all_file_list)*0.25))
train_file_list = [i for i in all_file_list if i not in test_file_list]

len(train_file_list), len(test_file_list)

dataset = load_dataset(
    "csv",
    data_files={
    'train':train_file_list,
    'valid':test_file_list
    },
    cache_dir="cache_data"
)
dataset

def get_masks_and_position_ids(
    seq, seq_len, context_length, device, gmask=False, position_encoding_2d=True
):
    mask_position = (
        seq_len - 2
    )  # is equal to `seq.index(mask_token)` or `seq.index(150001)`
    attention_mask = torch.ones((1, context_length, context_length), device=device)
    attention_mask.tril_()
    attention_mask[..., : mask_position - 1] = 1
    attention_mask = (attention_mask < 0.5).bool()

    if position_encoding_2d:
        seq_length = seq_len - 1  # is equal to `seq_length = seq.index(150004)`
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[seq_length:] = mask_position
        block_position_ids = torch.cat(
            (
                torch.zeros(seq_length, dtype=torch.long, device=device),
                torch.arange(
                    context_length - seq_length, dtype=torch.long, device=device
                )
                + 1,
            )
        )
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[context_length - 1 :] = mask_position
    return attention_mask, position_ids

def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids) + 1
    input_ids = []
    attention_mask_list = []
    position_ids_list = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
            [-100] * (seq_len - 1)
            + ids[(seq_len - 1) :]
            + [tokenizer.eop_token_id]
            + [-100] * (longest - ids_l - 1)
        )
        ids = ids + [tokenizer.eop_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        attention_mask, position_ids = get_masks_and_position_ids(
            ids, seq_len, longest, _ids.device, gmask=False
        )
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
        attention_mask_list.append(attention_mask)
        position_ids_list.append(position_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    attention_mask = torch.stack(attention_mask_list)
    position_ids = torch.stack(position_ids_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    # {"context": context, "target": target}
    example['context'] = context
    example['target'] = target
    return example

max_seq_length = 512

def preprocess(example):
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target, max_length=max_seq_length, truncation=True, add_special_tokens=False
    )
    input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}

def filter_nan(example):
    return example['target'] is not None and example['context'] is not  None


tokenized_datasets = dataset.map(
    function=format_example, remove_columns=dataset['train'].column_names
    ).filter(function=filter_nan)
tokenized_datasets = tokenized_datasets.map(function=preprocess)
tokenized_datasets

class EmptyCacheCallBack(TrainerCallback):
    """
    通过callback的形式，解决显存不够的问题

    """

    def __init__(self) -> None:
        super().__init__()

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs, **kwargs):
        """
        Event called after logging the last logs.
        """
        torch.cuda.empty_cache()

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        torch.cuda.empty_cache()

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        torch.cuda.empty_cache()
        
    

eccb = EmptyCacheCallBack()


args = TrainingArguments(
    output_dir="test004",
    per_device_train_batch_size=2, 
    per_device_eval_batch_size=1,
    evaluation_strategy="steps",
    eval_steps=50,
    logging_steps=50,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=100,
    fp16=True,
    push_to_hub=False,
    remove_unused_columns=False
)

trainer = MyTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    # callbacks=[eccb]
)
trainer.train()