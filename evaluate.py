# -*- coding: utf-8 -*-
import argparse
parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 models with multi-task SFT")
parser.add_argument('--cuda_device', type=str, default="1", help="CUDA device number(s) to use (e.g., '6' or '0,1')")
parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-0.5B-Instruct_sft_multi_task_16_2",
                   help="Model name or path from the available qwen_models list")
parser.add_argument('--dataset', type=str, default="test.jsonl",
                   help="Model name or path from the available qwen_models list")
parser.add_argument('--best_of', type=int, default=1, 
                   help="Number of output sequences to generate for each prompt (best result will be used)")
parser.add_argument('--n', type=int, default=1,
                   help="Number of top sequences to return from best_of generations")
parser.add_argument('--top_k', type=int, default=0,
                   help="Number of top generations to use for average F1 calculation (default: use all generations)")
parser.add_argument('--temperature', type=float, default=0.0)
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from datasets import load_dataset
import json
import re
from typing import Dict, List
from prompt_background import ner_background, re_background
from vllm import LLM, SamplingParams
import logging
from datetime import datetime
import sys
import pandas as pd

lora_path = args.model_name
load_in_4bit = '4bit' in lora_path
dataset_file = args.dataset
if not dataset_file.endswith('.jsonl'):
    print("Dataset file must end with '.jsonl' (e.g., test.jsonl)")
    sys.exit(1)
dataset_name = dataset_file.replace('.jsonl', '')

log_dir = f"logs/evaluategrpotemp/{lora_path}"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'eval_{dataset_name}_bestof{args.best_of}_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Replace the model loading code with vLLM
model = LLM(
    model=lora_path,
    dtype="bfloat16",
    gpu_memory_utilization=0.8,
    tensor_parallel_size=len(args.cuda_device.split(',')) if ',' in args.cuda_device else 1,
)

dataset_path = f"SciER/SciER/LLM/{dataset_file}"
dataset = load_dataset("json", data_files=dataset_path)["train"]

SYSTEM_PROMPT = """
    Respond in the following format:
    <reasoning>
    Provide step-by-step reasoning to solve the task based on the given instructions and sentence.
    </reasoning>
    <think>
    Cite the specific sentence part (e.g., phrase, verb, or structure) supporting the relation.
    Articulate a symbolic pattern you discovered (e.g., "The verb 'achieves' suggests a Method is applied to a Task, implying a relation").
    Explain how this pattern leads to the predicted relation, referencing the relationship definition.
    Use concise, logical chains (e.g., "X performs Y → relation Z because of definition").
    </think>
    <answer>
    Provide the final answer in JSON format as specified in the instruction.
    </answer>"""

def format_ner_re_end2end_prompt(example: Dict) -> Dict:
    sentence = example["sentence"]
    prompt = f"""{ner_background}

{re_background}

Given the sentence: "{sentence}"

Extract entities and their relations.

### Instruction:
- Think step-by-step to identify entities ('Dataset', 'Task', 'Method') and their relationships.
- Return the results in JSON format with:
  - "ner": a list of [entity, type] pairs.
  - "rel": a list of [subject, relation, object] triples.
"""
    # Modify to return just the prompt text instead of chat format
    return {
        "prompt": f"<s>[INST] {SYSTEM_PROMPT} [/INST]\n\n[INST] {prompt} [/INST]",
        "expected": {"ner": example["ner"], "rel": example["rel"]}
    }

eval_dataset = dataset.map(format_ner_re_end2end_prompt, batched=False)

sampling_params = SamplingParams(
    temperature=args.temperature,
    best_of=args.best_of,
    n=args.n,
    max_tokens=1024,
)

def extract_xml_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def compute_metrics(expected: List, predicted: List) -> Dict[str, float]:
    if not expected and not predicted:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not expected:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    def make_hashable(item):
        if isinstance(item, list):
            return tuple(make_hashable(i) for i in item)
        return item
    
    try:
        expected_set = set(tuple(make_hashable(x) for x in expected))
        predicted_set = set(tuple(make_hashable(x) for x in predicted))
    except TypeError as e:
        logger.error(f"Unhashable type in data: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    tp = len(expected_set & predicted_set)
    fp = len(predicted_set - expected_set)
    fn = len(expected_set - predicted_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

def evaluate_dataset(dataset, dataset_name: str, batch_size: int = 8):
    ner_metrics = {"tp": 0, "fp": 0, "fn": 0}
    rel_metrics = {"tp": 0, "fp": 0, "fn": 0}
    total_samples = len(dataset)
    
    prompts = [example["prompt"] for example in dataset]
    expected_outputs = [example["expected"] for example in dataset]

    outputs = model.generate(prompts, sampling_params)
    
    for i, (output, expected) in enumerate(zip(outputs, expected_outputs)):
        all_outputs_text = [gen_output.text for gen_output in output.outputs]
        
        best_ner_f1 = 0.0
        best_rel_f1 = 0.0
        best_predicted_ner = []
        best_predicted_rel = []
        
        expected_ner = expected["ner"]
        expected_rel = expected["rel"]
        
        for output_text in all_outputs_text:
            predicted_answer = extract_xml_answer(output_text)
            try:
                predicted_json = json.loads(predicted_answer) if predicted_answer else {"ner": [], "rel": []}
                
                if isinstance(predicted_json, list):
                    predicted_ner = []
                    predicted_rel = []
                    for item in predicted_json:
                        if isinstance(item, dict):
                            if "ner" in item:
                                predicted_ner.extend(item["ner"] if isinstance(item["ner"], list) else [])
                            if "rel" in item:
                                predicted_rel.extend(item["rel"] if isinstance(item["rel"], list) else [])
                else:
                    predicted_ner = predicted_json.get("ner", [])
                    predicted_rel = predicted_json.get("rel", [])
                    
                    if not isinstance(predicted_ner, list):
                        predicted_ner = []
                    if not isinstance(predicted_rel, list):
                        predicted_rel = []
                        
            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                predicted_ner = []
                predicted_rel = []
            
            ner_result = compute_metrics(expected_ner, predicted_ner)
            rel_result = compute_metrics(expected_rel, predicted_rel)
            
            if ner_result["f1"] > best_ner_f1:
                best_ner_f1 = ner_result["f1"]
                best_predicted_ner = predicted_ner
            
            if rel_result["f1"] > best_rel_f1:
                best_rel_f1 = rel_result["f1"]
                best_predicted_rel = predicted_rel
        
        ner_metrics["tp"] += len(set(tuple(x) for x in expected_ner) & set(tuple(x) for x in best_predicted_ner))
        ner_metrics["fp"] += len(set(tuple(x) for x in best_predicted_ner) - set(tuple(x) for x in expected_ner))
        ner_metrics["fn"] += len(set(tuple(x) for x in expected_ner) - set(tuple(x) for x in best_predicted_ner))
        
        rel_metrics["tp"] += len(set(tuple(x) for x in expected_rel) & set(tuple(x) for x in best_predicted_rel))
        rel_metrics["fp"] += len(set(tuple(x) for x in best_predicted_rel) - set(tuple(x) for x in expected_rel))
        rel_metrics["fn"] += len(set(tuple(x) for x in expected_rel) - set(tuple(x) for x in best_predicted_rel))

    ner_precision = ner_metrics["tp"] / (ner_metrics["tp"] + ner_metrics["fp"]) if (ner_metrics["tp"] + ner_metrics["fp"]) > 0 else 0.0
    ner_recall = ner_metrics["tp"] / (ner_metrics["tp"] + ner_metrics["fn"]) if (ner_metrics["tp"] + ner_metrics["fn"]) > 0 else 0.0
    ner_f1_micro = 2 * (ner_precision * ner_recall) / (ner_precision + ner_recall) if (ner_precision + ner_recall) > 0 else 0.0

    rel_precision = rel_metrics["tp"] / (rel_metrics["tp"] + rel_metrics["fp"]) if (rel_metrics["tp"] + rel_metrics["fp"]) > 0 else 0.0
    rel_recall = rel_metrics["tp"] / (rel_metrics["tp"] + rel_metrics["fn"]) if (rel_metrics["tp"] + rel_metrics["fn"]) > 0 else 0.0
    rel_f1_micro = 2 * (rel_precision * rel_recall) / (rel_precision + rel_recall) if (rel_precision + rel_recall) > 0 else 0.0

    print(f"{dataset_name} Evaluation Results:")
    print(f"NER - Precision: {ner_precision:.4f}, Recall: {ner_recall:.4f}, F1: {ner_f1_micro:.4f}")
    print(f"REL - Precision: {rel_precision:.4f}, Recall: {rel_recall:.4f}, F1: {rel_f1_micro:.4f}")

evaluate_dataset(
    eval_dataset, 
    dataset_name.capitalize(),
)