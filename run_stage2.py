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
    parser.add_argument('--judge_llm', type=str, default='meta-llama/Llama-3.3-70B-Instruct-Turbo', help='LLM for judge')
    parser.add_argument('--rater_llm', type=str, default='gpt-3.5-turbo-1106', help='LLM for rater')
    parser.add_argument('--num_mix_skill', type=int, default=1, help='Number of skills to be mixed')
    parser.add_argument('--num_sample_per_comb', type=int, default=20, help='Number of samples per combination')
    parser.add_argument('--root_dir', type=str, default='./output', help='Root Directory')
    parser.add_argument('--skill_path', type=str, default='./skills.csv', help='Path to Skill')
    parser.add_argument('--seed', type=int, default=952, help='Batch size')

    parser.add_argument('--openai_key', type=str, default='', help='OpenAI access key')
    parser.add_argument('--together_ai_key', type=str, default='', help='Together AI access key')

    return parser.parse_args()


def main():
    args = parse_args()
    
    work_dir = os.path.join(args.root_dir, f'IH_{args.target_llm}_skill{args.num_mix_skill}')
    os.makedirs(work_dir, exist_ok=True)
    if os.path.exists(os.path.join(work_dir, 'configs.json')):
        with open(os.path.join(work_dir, 'configs.json'), 'r') as file:
            configs = json.load(file)
    else:
        raise Exception("No stage 1 data found.")

    np.random.seed(args.seed)

    client = Together(api_key=args.together_ai_key)
    
    df_skills = pd.read_csv(args.skill_path)
    all_skills = df_skills["Skill"].tolist()[:2]
    
    jbb_dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    all_intents = jbb_dataset['harmful']['Goal'][:2]
    
    llm = jbb.LLMLiteLLM(model_name=args.target_llm, api_key=args.openai_key)  # target model
    
    conditioned_judge = GPTConditionedJudge(api_key=args.openai_key)  # 70B judge model:contentReference[oaicite:12]{index=12}
    jailbreak_judge = Llama3JailbreakJudge(api_key=args.together_ai_key)  
    refusal_judge = Llama3RefusalJudge(api_key=args.together_ai_key) 
    
    attack_samples = construct_data_stage2(args, configs, refusal_judge, client, all_skills, all_intents, args.num_mix_skill, num_sample_per_comb=args.num_sample_per_comb)
    configs['attack_samples_stage2'] = attack_samples
    
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
    configs['target_rejects_stage2'] = target_rejects
    configs['attack_samples_stage2'] =  new_attack_examples
    
    # Evaluation based on LLM judge
    results = configs['attack_samples_stage2']
    chunk_size = 20
    evals = []
    for i in tqdm(range(0, len(results), chunk_size)):
        curr_results = results[i:min(i+chunk_size, len(results))]
        # Query the target model to get its response to the attack prompt
        prompts = [result["prompt"] for result in curr_results]
        responses = [result["response"] for result in curr_results]
        for j, result in enumerate(curr_results):
            result['response'] = responses[j]
            curr_results[j] = result
        
        classifications = jailbreak_judge(prompts=prompts, responses=responses)  # assuming the judge can output structured JSON
       
        for k, result in enumerate(curr_results):
            evals.append({
                "prompt": result["prompt"],
                "intent": result["intent"],
                "skills": result["skills"],
                "response": result['response'],
                'classification': classifications[k],
                #'score': score[k]
            })
    configs['evals_clas_stage2'] = evals

    # Evaluation based on LLM conditioned judge
    evals = []
    for i in tqdm(range(0, len(results), chunk_size)):
        curr_results = results[i:min(i+chunk_size, len(results))]
        # Query the target model to get its response to the attack prompt
        prompts = [result["prompt"] for result in curr_results]
        responses = [result["response"] for result in curr_results]
        for j, result in enumerate(curr_results):
            result['response'] = responses[j]
            curr_results[j] = result
        
        scores = conditioned_judge(curr_results)
       
        for k, result in enumerate(curr_results):
            evals.append({
                "prompt": result["prompt"],
                "intent": result["intent"],
                "skills": result["skills"],
                "response": result['response'],
                'score': scores[k],
                #'score': score[k]
            })
    configs['evals_conditioned_stage2'] = evals
    save_config(configs, work_dir)

    data_types = ['intent', 'skills', 'intent_skills']
    
    hybrid_result = compute_hybrid_scores(configs, data_types=data_types, all_intents=all_intents, use_binary_score=True)
    global_score = hybrid_result['global']
    print(f'Binary JR Score: {global_score}')
    hybrid_result = compute_hybrid_scores(configs, data_types=data_types, all_intents=all_intents, use_binary_score=False)
    global_score = hybrid_result['global']
    print(f'JR Score: {global_score}')

if __name__ == '__main__':
    main()