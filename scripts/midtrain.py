import transformers
from transformers import MistralConfig, MistralForCausalLM, TrainingArguments, Trainer, HfArgumentParser, MixtralConfig, MixtralForCausalLM
from dataclasses import dataclass, field
from typing import Optional
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from model import AminoAcidTokenizer
import torch
from glob import glob
import pandas as pd
from datasets import load_dataset
import os
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

def preprocess(sources, tokenizer):
    input_ids, labels = [], []
    for source in tqdm(sources):
        inputs = tokenizer(source, return_tensors='pt', truncation=True, max_length=2048)
        ids = inputs['input_ids'][0]
        input_ids.append(ids)
        labels.append(ids.clone())

    return {
        'input_ids': input_ids,
        'labels': labels
    }

class AntibodyDesignDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        super(AntibodyDesignDataset, self).__init__()
        sources = [example['whole_seq'] for  _, example in data.iterrows()]
        data_dict = preprocess(sources, tokenizer)
        self.input_ids = data_dict['input_ids']
        self.labels = data_dict['labels']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return dict(
            input_ids=self.input_ids[idx],
            labels=self.labels[idx],
        )


class AntibodyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]

        print(f'pad token id is {tokenizer.pad_token_id}')
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 表示 loss 忽略位置

        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id).long(),
            "labels": labels
        }


@dataclass
class ModelArguments:
    config_path: Optional[str] = field(default="", metadata={"help": "Your model name"})
    model_path: Optional[str] = field(default="",metadata={"help":"ckpt path"})
@dataclass
class DataArguments:
    data_path: Optional[str] = field(default='', metadata={"help": "Your data path"})



print('Start main program')
parser = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments))
training_args,model_args, data_args = parser.parse_args_into_dataclasses()
training_args.remove_unused_columns = False

if 'mistral' in model_args.config_path:
    config = MistralConfig.from_pretrained(model_args.model_path)
    print('mistral model pretrained')
    config._attn_implementation = 'eager'
    model = MistralForCausalLM.from_pretrained(model_args.model_path,config=config, torch_dtype=torch.bfloat16)
elif 'mixtral' in model_args.config_path:
    config = MixtralConfig.from_json_file(model_args.config_path)
    model = MixtralForCausalLM(config)
else:
    raise NotImplementedError


complex_data = pd.read_parquet(data_args.data_path)
print(len(complex_data))
tokenizer = AminoAcidTokenizer.from_pretrained('model/tokenizer')
dataset = AntibodyDesignDataset(complex_data, tokenizer)
collator = AntibodyCollator(tokenizer)

print(model.config)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=collator,
    train_dataset=dataset,
)

trainer.train()