import pandas as pd
import json
import os
import gzip
from glob import glob
from tqdm import tqdm
import numpy as np

root = "data/asd/asd"
file_paths = glob(os.path.join(root, "*.parquet"))
df = pd.concat([pd.read_parquet(fp) for fp in file_paths], ignore_index=True)

df = df[df['scfv']==False].reset_index(drop=True)
df = df[df['nanobody']==False].reset_index(drop=True)
df = df[df['confidence'].isin(['very_high','high'])].reset_index(drop=True) #776603
df = df[df['light_sequence'].notna()].reset_index(drop=True) # 691864
print(len(df))
df = df[df['heavy_sequence'].notna()].reset_index(drop=True)
print(len(df))
df = df[['heavy_sequence','light_sequence','antigen_sequence','affinity_type','affinity']]
df['whole_seq'] = df['antigen_sequence']+df['heavy_sequence']+df['light_sequence']
df.to_parquet("data/asd/complex.parquet")

