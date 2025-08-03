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
import os

class ABDataset(Dataset):
    def __init__(self,sequences, tokenizer):
        self.sequences = sequences
    
    def __getitem__(self,idx):
        return self.sequences[idx]
    
    def __len__(self):
        return len(self.sequences)

class ABCollator:
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer

    def __call__(self,batch):
        tokenized = [self.tokenizer.encode(seq, return_tensors='pt', add_special_tokens=True).squeeze(0) for seq in batch]
        
        pad_token_id = tokenizer._convert_token_to_id(tokenizer.pad_token)
        input_ids = pad_sequence(tokenized, batch_first=True, padding_value=pad_token_id)
        
        attention_mask = (input_ids != pad_token_id).long()
        
        labels = input_ids.clone()
        labels[input_ids == pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

@dataclass
class ModelArguments:
    config_path: Optional[str] = field(default="", metadata={"help": "Your model name"})

@dataclass
class DataArguments:
    data_path: Optional[str] = field(default='', metadata={"help": "Your data path"})



print('Start main program')
parser = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments))
training_args,model_args, data_args = parser.parse_args_into_dataclasses()
training_args.remove_unused_columns = False

if 'mistral' in model_args.config_path:
    config = MistralConfig.from_json_file(model_args.config_path)
    model = MistralForCausalLM(config)
elif 'mixtral' in model_args.config_path:
    config = MixtralConfig.from_json_file(model_args.config_path)
    model = MixtralForCausalLM(config)
else:
    raise NotImplementedError

model.config._attn_implementation = 'eager'

sequences = []
for file in glob(os.path.join(data_args.data_path, '*.parquet')):
    df = pd.read_parquet(file, columns=['sequence'])
    sequences.extend(df['sequence'].tolist())

tokenizer = AminoAcidTokenizer.from_pretrained('model/tokenizer')
dataset = ABDataset(sequences, tokenizer)
collator = ABCollator(tokenizer)

print(model.config)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=collator,
    train_dataset=dataset,
)

trainer.train()