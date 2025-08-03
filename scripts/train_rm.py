import transformers
from transformers import MistralConfig, MistralForSequenceClassification, TrainingArguments, Trainer, HfArgumentParser, PreTrainedTokenizer
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
import math
from typing import Dict, Optional
from model import AminoAcidTokenizer
from tqdm import tqdm


def preprocess(
    sources,
    tokenizer,
    max_len,
    is_train=True,
):
    input_ids, labels = [], []

    if is_train:
        percent = float(os.getenv("PERCENT_DATA", "1"))
        print(f"Using {percent} of the data for training.")
        sources = sources[:int(len(sources) * percent)]
        print(f"Using {len(sources)} samples for training.")
    
    for i, source in tqdm(enumerate(sources)):
        label = source[1]
        af_type = source[2]
        seq = source[0]
        input_id = tokenizer.encode(seq, max_length=max_len, truncation=True)
        input_ids.append(input_id)
        labels.append(label)
    
    print(f"data size: {len(input_ids)}")
    return dict(
        input_ids=input_ids,
        labels=labels,
    )
    
    
class SupervisedDataset(Dataset):
    def __init__(self, raw_data, tokenizer: PreTrainedTokenizer, max_len: int, is_train=True):
        super(SupervisedDataset, self).__init__()
        sources = [(example['whole_seq'], example['affinity'], example['affinity_type']) for  _, example in raw_data.iterrows()]
        data_dict = preprocess(sources, tokenizer, max_len, is_train)
        self.input_ids = data_dict['input_ids']
        self.labels = data_dict['labels']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return dict(
            input_ids=self.input_ids[idx],
            labels=self.labels[idx],
        )

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    # train_parquet = load_dataset("parquet",data_args.data_path)
    df = pd.read_parquet(data_args.data_path)
    # df.to_dict(orient="records")
    
    train_dataset = SupervisedDataset(df, tokenizer=tokenizer, max_len=max_len)

    # eval_parquet = load_dataset(data_args.data_path)['validation']
    # eval_dataset = SupervisedDataset(eval_parquet, tokenizer=tokenizer, max_len=max_len, is_train=False)

    return dict(train_dataset=train_dataset)


from torch.nn.utils.rnn import pad_sequence
class CustomDataCollatorForClassification:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):

        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        labels = torch.tensor([item['labels'] for item in batch])

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return {
            'input_ids': input_ids,
            'labels': labels.bfloat16(),
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id).long(),
        }

@dataclass
class ModelArguments:
    config_path: Optional[str] = field(default="", metadata={"help": "Your model name"})
    model_name_or_path: Optional[str] = field(default="",metadata={"help":"ckpt path"})
@dataclass
class DataArguments:
    data_path: Optional[str] = field(default='', metadata={"help": "Your data path"})



print('Start main program')
parser = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments))
training_args,model_args, data_args = parser.parse_args_into_dataclasses()
training_args.remove_unused_columns = False

config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
config.use_cache = False
config._attn_implementation = 'eager'
tokenizer = AminoAcidTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
config.pad_token_id = tokenizer.pad_token_id
config.num_labels = 1

print('mistral model pretrained')
model = MistralForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    torch_dtype=torch.bfloat16,
    config=config,
    trust_remote_code=True,
    )

data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=3000
    )
data_collator = CustomDataCollatorForClassification(tokenizer)


trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    **data_module
)
trainer.train()
trainer.save_state()