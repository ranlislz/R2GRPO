import os
# Set environment variables and configurations based on arguments
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import json
import re
from typing import Dict, List
from prompt_background import ner_background, re_background, re_golden_background, re_plus_background, ner_background_erc, re_background_erc

# Argument parser setup
parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 models with multi-task SFT")
parser.add_argument('--cuda_device', type=str, default="4", help="CUDA device number(s) to use (e.g., '6' or '0,1')")
parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-0.5B-Instruct_sft_multi_task_16_2",
                   help="Model name or path")
parser.add_argument('--tasks', type=str, nargs='+', default=["end2end"],
                   choices=["ner", "re", "re_plus", "end2end"],
                   help="Tasks to train on: ner, re, re_plus, end2end (space-separated)")
parser.add_argument('--train_epochs', type=int, default=3,
                   )
args = parser.parse_args()

# Model configuration
max_seq_length = 2500
train_epochs = args.train_epochs
per_device_train_batch_size = 24

lora_rank = 16
lora_alpha = 32

# Validate and set model_name
model_name = args.model_name
load_in_4bit = '4bit' in model_name
dtype = torch.bfloat16

# Task selection based on arguments
SELECTED_TASKS = args.tasks
print(f"Selected tasks for SFT: {SELECTED_TASKS}")

# Load and split the dataset
task_name = '_'.join(sorted(args.tasks))  # 'ner_re' from ['ner', 're']
dataset_name = ''
# dataset_name = ''
dataset = load_dataset("json",
                       data_files={"train": f"SciER/SciER/LLM/train{dataset_name}.jsonl"})
train_val_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)  # 90% train, 10% val
train_dataset = train_val_dataset["train"]
# train_dataset = dataset["train"]
val_dataset = train_val_dataset["test"]
# test_dataset = dataset["test"]
length = len(train_dataset)*len(args.tasks)
# *len(args.tasks)
max_steps = length // per_device_train_batch_size * train_epochs

# Background information for prompts
if 'erc' in dataset_name:
    ner_background = ner_background_erc
    re_background = re_background_erc
else:
    ner_background = ner_background
    re_background = re_background
re_plus_background = re_plus_background

# System prompt for reasoning and format
SYSTEM_PROMPT = """
Respond in the following format:
<plan>
Provide step-by-step plan to solve the task based on the given instructions and sentence.
</plan>
<answer>
Provide the final answer in JSON format as specified in the instruction.
</answer>
"""

# Prompt formatting functions
def format_ner_prompt(example: Dict) -> Dict:
    sentence = example["sentence"]
    ner = example["ner"]

    prompt = f"""{ner_background}

Given the sentence: "{sentence}"

Extract the named entities and their types.

### Instruction:
- Think step-by-step to identify entities of type 'Dataset', 'Task', and 'Method'.
- Return the results in JSON format with:
  - "ner": a list of [entity, type] pairs.
"""

    response_dict = {"ner": ner}
    response = f"""<plan>
To extract named entities from the sentence "{sentence}", I will:
1. Read the sentence carefully to identify potential entities.
2. Check each phrase against the definitions in the instruction.
3. Ensure that only factual, content-bearing entities are annotated, excluding generics and determiners.
4. Validate that the identified entities match the expected types based on the context.
</plan>
<answer>
{json.dumps(response_dict, ensure_ascii=False)}
</answer>"""
    convo = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    return {"conversations": convo, "task": "ner"}

def format_re_prompt(example: Dict) -> Dict:
    sentence = example["sentence"]
    rel = example["rel"]
    ner_dict = {entity[0]: entity[1] for entity in example["ner"]}
    entity_pairs = [f"{s} ({ner_dict.get(s, 'Unknown')}) - {o} ({ner_dict.get(o, 'Unknown')})" for s, _, o in rel]
    pairs_str = "\n".join(entity_pairs) if entity_pairs else "No entity pairs provided."

    prompt = f"""{ner_background}
    
    {re_background}

Given the sentence: "{sentence}"

And the following subject-object entity pairs with their types:
{pairs_str}

Determine the relationship between each subject and object pair based on the sentence.

### Instruction:
- Think step-by-step to determine the relationship between each pair.
- Return the results in JSON format with:
  - "rel": a list of [subject, relation, object] triples.
"""

    response_dict = {"rel": rel}
    response = f"""<plan>
To determine relationships in the sentence "{sentence}":
1. Analyze the sentence context and the provided entity pairs.
2. For each pair, evaluate the potential relationships based on the definitions.
3. Ensure the relationship is directly supported by the sentence, avoiding speculative inferences.
4. Compile the valid relationships into a list of triples.
</plan>
<answer>
{json.dumps(response_dict, ensure_ascii=False)}
</answer>"""
    convo = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    return {"conversations": convo, "task": "re"}

def format_re_plus_prompt(example: Dict) -> Dict:
    sentence = example["sentence"]
    ner = example["ner"]
    rel_plus = example["rel_plus"]
    entities_str = "\n".join([f"{entity} ({type_})" for entity, type_ in ner]) if ner else "No entities provided."

    prompt = f"""{re_plus_background}

Given the sentence: "{sentence}"

And the following list of extracted entities with their types:
{entities_str}

Extract relationship triplets based on the sentence and the provided entities.

### Instruction:
- Think step-by-step to identify relationships between the provided entities.
- Return the results in JSON format with:
  - "rel_plus": a list of [entity1:type, relation, entity2:type] triples, or "rel_plus": "NULL" if there are no triplets.
"""

    response_dict = {"rel_plus": rel_plus if rel_plus else "NULL"}
    response = f"""<plan>
To extract relationship triplets from the sentence "{sentence}":
1. Review the provided list of entities: {entities_str}.
2. Analyze the sentence context to identify potential relationships between these entities.
3. For each pair of entities, determine if a relationship exists from the list of potential relations.
4. Form triplets in the format [entity1:type, relation, entity2:type] based on clear evidence in the text.
5. If no triplets are found, return "NULL".
</plan>
<answer>
{json.dumps(response_dict, ensure_ascii=False)}
</answer>"""
    convo = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    return {"conversations": convo, "task": "re_plus"}

def format_end2end_prompt(example: Dict) -> Dict:
    sentence = example["sentence"]
    ner = example["ner"]
    rel = example["rel"]

    prompt = f"""{ner_background}

{re_background}

Given the sentence: "{sentence}"

Extract entities and their relations.

### Instruction:
- Think step-by-step to identify entities and their relationships.
- Return the results in JSON format with:
  - "ner": a list of [entity, type] pairs.
  - "rel": a list of [subject, relation, object] triples.
"""

    response_dict = {"ner": ner, "rel": rel}
    response = f"""<plan>
To perform end-to-end extraction from the sentence "{sentence}":
1. Identify entities and their types based on definitions and instruction.
2. Enumerate all the possible entity pairs to infer valid relations between them.
3. Cite the specific sentence part with ""(e.g., phrase, verb, or structure) supporting the relation.
4. Articulate the symbolic pattern you discovered (e.g., "The verb 'achieves' suggests a Method is applied to a Task, implying a relation").
5. Explain how this pattern leads to the predicted relation using concise, logical chains (e.g., "X performs Y → relation Z because of definition"). 
</plan>
<answer>
{json.dumps(response_dict, ensure_ascii=False)}
</answer>"""
    convo = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    return {"conversations": convo, "task": "end2end"}

# Prepare multi-task dataset with task selection
def prepare_multi_task_dataset(dataset: Dataset) -> Dataset:
    combined_data = []
    
    if "ner" in SELECTED_TASKS:
        ner_examples = dataset.map(format_ner_prompt, batched=False)
        combined_data.extend(ner_examples)
        print(f"Added NER task with {len(ner_examples)} examples")
    
    if "re" in SELECTED_TASKS:
        re_examples = dataset.map(format_re_prompt, batched=False)
        combined_data.extend(re_examples)
        print(f"Added RE task with {len(re_examples)} examples")
    
    if "re_plus" in SELECTED_TASKS:
        re_plus_examples = dataset.map(format_re_plus_prompt, batched=False)
        combined_data.extend(re_plus_examples)
        print(f"Added RE+ task with {len(re_plus_examples)} examples")
    
    if "end2end" in SELECTED_TASKS:
        end2end_examples = dataset.map(format_end2end_prompt, batched=False)
        combined_data.extend(end2end_examples)
        print(f"Added End2End task with {len(end2end_examples)} examples")

    if not combined_data:
        raise ValueError("No tasks selected for training. Please include at least one task in SELECTED_TASKS.")
    
    return Dataset.from_list(combined_data)

# Prepare train and validation datasets
train_multi_task_dataset = prepare_multi_task_dataset(train_dataset)
val_multi_task_dataset = prepare_multi_task_dataset(val_dataset)

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # gpu_memory_utilization = 0.15,
    fast_inference=True,
    local_files_only=True,
)

# Apply LoRA for efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_alpha,
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Set up chat template for Qwen-2.5
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
)
if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# Apply chat template to datasets and extract response part for training
def extract_response(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else text

train_multi_task_dataset = train_multi_task_dataset.map(
    lambda x: {
        "text": tokenizer.apply_chat_template(x["conversations"], tokenize=False, add_generation_prompt=False),
        "response": extract_response(x["conversations"][2]["content"])
    },
    batched=False,
    desc="Applying chat template to train dataset"
)
val_multi_task_dataset = val_multi_task_dataset.map(
    lambda x: {
        "text": tokenizer.apply_chat_template(x["conversations"], tokenize=False, add_generation_prompt=False),
        "response": extract_response(x["conversations"][2]["content"])
    },
    batched=False,
    desc="Applying chat template to val dataset"
)

# Verify dataset structure
print("Sample from train dataset:", train_multi_task_dataset[0])

# Training setup
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_multi_task_dataset,
    eval_dataset=val_multi_task_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        warmup_steps=int(0.07 * max_steps),
        num_train_epochs=train_epochs,
        learning_rate=1e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=int(0.02 * max_steps),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=f"loras/sftplan//{model_name}_{task_name}_outputs_new{dataset_name}",
        report_to="tensorboard",
        evaluation_strategy="steps",
        eval_steps=int(0.1 * max_steps),
        save_strategy="steps",
        save_steps=int(0.1 * max_steps),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    ),
)

# Train only on assistant responses
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

# Train the model
trainer_stats = trainer.train()

# Save the LoRA adapters
model.save_pretrained(f"loras/sftplan/{model_name}_sft_{task_name}_{lora_rank}_{train_epochs}_plan_new{dataset_name}")
tokenizer.save_pretrained(f"loras/sftplan/{model_name}_sft_{task_name}_{lora_rank}_{train_epochs}_plan_new{dataset_name}")

# Merge and save the model in 16-bit format
model.save_pretrained_merged(
    f"merged_models/sftplan/{model_name}_sft_{task_name}_{lora_rank}_{train_epochs}_plan_new{dataset_name}",
    tokenizer,
    save_method="merged_16bit",
)

print(f"Training completed. LoRA saved to loras/sftplan/{model_name}_sft_{task_name}_{lora_rank}_{train_epochs}_plan_new{dataset_name}")
print(f"Merged 16-bit model saved to merged_models/sftplan/{model_name}_sft_{task_name}_{lora_rank}_{train_epochs}_plan_new{dataset_name}")