import jailbreakbench as jbb
from datasets import load_dataset
from openai import OpenAI
from together import Together
import pandas as pd
import itertools
import numpy as np
import json
import concurrent.futures
import collections
import os
import pdb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import *
from models import *

import argparse
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Example script that saves args to config.json")

    parser.add_argument('--target_llm', type=str, default='gpt-3.5-turbo-1106', help='Target LLM')
    parser.add_argument('--E_llm', type=str, default='meta-llama/Llama-3.3-70B-Instruct-Turbo', help='LLM for model E')
    parser.add_argument('--num_mix_skill', type=int, default=1, help='Number of skills to be mixed')
    parser.add_argument('--num_sample_per_comb', type=int, default=5, help='Number of samples per combination')
    parser.add_argument('--root_dir', type=str, default='./output', help='Root Directory')
    parser.add_argument('--skill_path', type=str, default='./skills.csv', help='Path to Skill')
    #parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save outputs')
    parser.add_argument('--seed', type=int, default=952, help='Batch size')

    parser.add_argument('--openai_key', type=str, default='', help='OpenAI access key')
    parser.add_argument('--together_ai_key', type=str, default='', help='Together AI access key')

    return parser.parse_args()


def main():
    args = parse_args()
    
    work_dir = os.path.join(args.root_dir, f'IH_{args.target_llm}_skill{args.num_mix_skill}')
    os.makedirs(work_dir, exist_ok=True)
    configs = {}

    np.random.seed(args.seed)

    client = Together(api_key=args.together_ai_key)
    
    df_skills = pd.read_csv(args.skill_path)
    all_skills = df_skills["Skill"].tolist()[:2]
    
    jbb_dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    all_intents = jbb_dataset['harmful']['Goal'][:2]
    
    llm = jbb.LLMLiteLLM(model_name=args.target_llm, api_key=args.openai_key)  # target model
    refusal_judge = Llama3RefusalJudge(api_key=args.together_ai_key) 
    
    attack_samples = construct_data(args, refusal_judge, client, all_skills, all_intents, args.num_mix_skill, num_sample_per_comb=args.num_sample_per_comb)
    configs['attack_samples'] = attack_samples
    
    chunk_size = 20
    new_attack_examples = []
    target_rejects = []
    for i in tqdm(range(0, len(attack_samples), chunk_size)):
        samples = attack_samples[i:min(i+chunk_size, len(attack_samples))]
        # Query the target model to get its response to the attack prompt
        prompts = [sample["prompt"] for sample in samples]
        responses = llm.query(prompts=prompts).responses
        
        target_rejects.extend(refusal_judge(prompts=prompts, responses=responses))
        # Prepare input for the judge, including intent and skill context
           
        for j, sample in enumerate(samples):
            sample['response'] = responses[j]
            samples[j] = sample
        new_attack_examples.extend(samples)
    configs['target_rejects'] = target_rejects
    configs['attack_samples'] =  new_attack_examples
    
    
    target_rejects_dict = collections.defaultdict(list)
    for i, sample in enumerate(configs['attack_samples']):
        intent, skill = sample['intent'], str(sample['skills'])
        target_rejects_dict[f'{intent}_{skill}'].append(1-int(configs['target_rejects'][i]))
    for k, v in target_rejects_dict.items():
        #pdb.set_trace()
        target_rejects_dict[k] = float(np.mean(v))
    
    compute_extreme_skill_intent(target_rejects_dict, operation='max')
    configs['target_rejects_dict'] = target_rejects_dict
    save_config(configs, work_dir)


if __name__ == '__main__':
    main()