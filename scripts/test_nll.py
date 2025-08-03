from transformers import MistralForCausalLM, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from model import AminoAcidTokenizer
import torch
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
import numpy as np
import os
from Bio.SeqUtils import seq1
import json
from collections import defaultdict
import argparse

def fsdp2hf(fsdp_checkpoint_path):

    config = AutoConfig.from_pretrained(os.path.join(fsdp_checkpoint_path,'huggingface'))

    state_dict = defaultdict(list)
    world_size = 4
    for rank in range(world_size):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        print('loading', filepath)
        this_state_dict = torch.load(filepath)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())
    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'model parameters: {total_params}')

    tokenizer = AminoAcidTokenizer.from_pretrained(os.path.join(fsdp_checkpoint_path,'huggingface'))

    return model,tokenizer



def extract_sequences_from_pdb(pdb_file, chains):
    """
    Extracts chain sequences from a PDB, returning (heavy_seq, light_seq, antigen_seq)
    for the chains specified in the order [heavy, light, *antigen(s)].
    """
    from Bio import PDB
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("PDB", pdb_file)

    heavy_chain = chains[0] if len(chains) > 0 else None
    light_chain = chains[1] if len(chains) > 1 else None
    antigen_chains = chains[2:] if len(chains) > 2 else []

    heavy_seq, light_seq, antigen_seq = "", "", ""

    for model in structure:
        for chain in model:
            seq = []
            for residue in chain:
                if residue.get_resname() in PDB.Polypeptide.standard_aa_names:
                    seq.append(seq1(residue.get_resname()))
            chain_seq = "".join(seq)

            if chain.id == heavy_chain:
                heavy_seq = chain_seq
            elif chain.id == light_chain:
                light_seq = chain_seq
            elif chain.id in antigen_chains:
                antigen_seq += chain_seq  # Concatenate multiple antigen chains

    return heavy_seq, light_seq, antigen_seq


def log_likelihood_batch(seq, model, tokenizer):

    inputs = tokenizer(seq,return_tensors='pt', truncation=True).to('cuda')
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        ll_scores = []
        
        input_ids = input_ids.view(-1)
        for i, token_idx in enumerate(input_ids):
            p_i = probs[0, i, token_idx]
            if p_i <= 0:
                ll_scores.append(float("-inf"))
            else:
                ll_scores.append(float(torch.log(p_i)))
                
        ll = np.mean(ll_scores)

    return ll


def log_likelihood_cdr(seq, cdr_start, cdr_end):
    inputs = tokenizer(seq, return_tensors='pt', truncation=True).to('cuda')
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

        input_ids = input_ids.view(-1)
        ll_scores = []
        for i in range(cdr_start, cdr_end):
            token_idx = input_ids[i]
            p_i = probs[0, i, token_idx]
            if p_i <= 0:
                ll_scores.append(float("-inf"))
            else:
                ll_scores.append(float(torch.log(p_i)))

        ll = np.mean(ll_scores)

    return ll

def benchmarkDiffab():
    
    name = "absci_her2_sc"
    folder = 
    prt = pd.read_csv(f'{folder}/{name}_parent.csv')
    wtH = prt.loc[0]["Heavy"]
    wtL = prt.loc[0]["Light"]
    CDR3 = prt.loc[0]["HCDR3"]
    CDR2 = prt.loc[0]["HCDR2"]
    CDR1 = prt.loc[0]["HCDR1"]
    CDR = [CDR1, CDR2, CDR3]
    start = []
    end = []

    h,l,ag = extract_sequences_from_pdb(f'{folder}/{name}.pdb',["B","A","C"])
    mutants = pd.read_csv(f'{folder}/{name}.csv')
    # mutants = mutants[mutants["Binder"]==True]
    for i in range(3):
        start_idx = wtH.find(CDR)
        end_idx = start_idx + len(CDR)
        start.append(start_idx)
        end.append(end_idx)
    assert (start[0]<start[1] & start[1]<start[2])
    new_H = []
    results = []
    for new_cdr in tqdm(mutants[["HCDR3, HCDR2, HCDR1"]]):
        heavy = ""
        heavy = wtH[:start[0]]+new_cdr["HCDR1"]+wtH[end[0]:start[1]]+new_cdr["HCDR2"] +wtH[end[1]:start[2]]+new_cdr["HCDR3"]+wtH[end[2]:]
        # heavy = pre + new_cdr + post
        new_H.append(heavy)
        # whole_seq = ag + heavy + wtL
        # new_start = len(ag) +start_idx
        # new_end = new_start +len(new_cdr)
        new_start = len(pre)
        new_end = new_start+len(new_cdr)
        # nll = log_likelihood_cdr(heavy, new_start, new_end)
        nll = log_likelihood_batch(ag+heavy+wtL)
        results.append(nll)

    mutants["nll"] = results
    mutants["newH"] = new_H
    out_csv = os.path.join(folder, f"model_scores_allmut.csv")
    mutants.to_csv(out_csv, index=False)
    binding_score = mutants["KD (nM)"]
    nll = mutants["nll"]
    corr = pd.concat([binding_score, nll], axis=1).corr(method='spearman').iloc[0, 1]
    # rho, p_value = spearmanr(df["nll"], df["binding_score"])
    print(f"{name}: Spearman 相关系数 = {corr:.4f}")

        

def AbBi(model, tokenizer, stage):
    folder_path = "AbBi"
    meta_file =
    # pdbnames = ['1mhp','1mlc', '1n8z', '2fjg', '3gbn', '4fqi', 'aayl49','aayl49_ML', 'aayl51']
    pdbnames = ['1mhp','1n8z', '3gbn']
    with open(meta_file, "r") as f:
        metadata = json.load(f)

    rho_list = []
    p_list = []

    for name in pdbnames:

        meta_info = metadata[name]

        pdb_path = meta_info["pdb_path"]
        heavy_chain = meta_info["heavy_chain"]
        light_chain = meta_info["light_chain"]
        antigen_chains = meta_info["antigen_chains"]
        affinity_data_files = meta_info['affinity_data']
        pdb_path = os.path.join('data/ABbi/complex_structure',pdb_path)

        wt_hc, wt_lc, wt_ag = extract_sequences_from_pdb(
            pdb_file=pdb_path,
            chains=[heavy_chain, light_chain] + antigen_chains
        )

        for affinity_data in affinity_data_files:
            results = []
            
            file_name = os.path.join('data/ABbi/binding_affinity', affinity_data)
            df = pd.read_csv(file_name)

            if "mut_heavy_chain_seq" not in df.columns or "binding_score" not in df.columns:
                print(f"[Pass] {file_name} missing column mut_heacy_chain_seq")
                continue

            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Scoring {affinity_data}"):
                mutated_heavy_chain_seq = row["mut_heavy_chain_seq"]
                # Basic check
                if not isinstance(mutated_heavy_chain_seq, str) or len(mutated_heavy_chain_seq) == 0:
                    results.append(None)
                    continue

                mutated_complex_seq =  wt_ag + mutated_heavy_chain_seq + wt_lc

                log_likelihoods = log_likelihood_batch(mutated_complex_seq, model, tokenizer)
                results.append(log_likelihoods)

            # 计算 spearman 相关
            df["nll"] = results

            out_basename = os.path.splitext(os.path.basename(affinity_data))[0]
            out_csv = os.path.join('data/ABbi', f"{out_basename}_{stage}_scores.csv")
            df.to_csv(out_csv, index=False)
            binding_score = df["binding_score"]
            nll = df["nll"]
            corr = pd.concat([binding_score, nll], axis=1).corr(method='spearman').iloc[0, 1]
            # rho, p_value = spearmanr(df["nll"], df["binding_score"])
            print(f"{file_name}: Spearman 相关系数 = {corr:.4f}")

            rho_list.append(corr)

    mean_rho = sum(rho_list) / len(rho_list)

    print("\n======= 汇总结果 =======")
    print(f"平均 Spearman 相关系数: {mean_rho:.4f}")


if __name__ =="__main__":
    fsdp_checkpoint_path = 
    model, tokenizer = fsdp2hf(fsdp_checkpoint_path)

    model = model.cuda()
    model.eval()
    AbBi(model, tokenizer, "penalty")

