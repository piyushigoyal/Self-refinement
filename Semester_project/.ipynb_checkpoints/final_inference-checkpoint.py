# %%
from unsloth import FastLanguageModel
import torch
import pandas as pd
import numpy as np
import random
import datasets
import re
import gc 
import argparse
import os
import time
from vllm import LLM, SamplingParams
import evaluate
from statistics import mean
from evaluate import load
from unsloth.chat_templates import get_chat_template
from vllm import LLM, SamplingParams
from transformers import AutoModel, AutoTokenizer
pd.set_option('display.max_colwidth', None)  # None means unlimited width
# Load the BERTScore evaluation metric
bertscore = evaluate.load("bertscore")

# %%
MAX_SEQ_LENGTH = 1024 # Choose any! We auto support RoPE Scaling internally!
DTYPE = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
LOAD_IN_4BIT = False # Use 4bit quantization to reduce memory usage. Can be False.
DATA_PATH = "/cluster/home/pgoyal/main/test/COT/cot_test.csv"

#a_b means a is the asker model and b is the ALL model
OUTPUT_PATH = "/cluster/home/pgoyal/main/test/predictions_COT"
# checkpoint_path = "/cluster/project/sachan/piyushi/merged_models_COT/gemma_2b/checkpoint-360"
# checkpoint_COT_all = "/cluster/project/sachan/piyushi/merged_models_all/gemma_2b/checkpoint-308"
# ASKER_PREDS = "/cluster/project/sachan/piyushi/asker_predictions/gemma2_9b/"

checkpoint_path = "/cluster/project/sachan/piyushi/merged_models_COT/qwen_1.5/checkpoint-708"
checkpoint_COT_all = "/cluster/project/sachan/piyushi/merged_models_all/qwen_1.5/checkpoint-608"
ASKER_PREDS = "/cluster/project/sachan/piyushi/asker_predictions/qwen_0.5/"

SEED = 42
random.seed(SEED)

# %% [markdown]
# ### Preprocess data

# %%
def prepare_test_data(path=DATA_PATH, remove_test_duplicates=True):
    # Load test data
    df_test = pd.read_csv(path)
    df_test.rename(columns={'answer':'response'}, inplace=True)
    
    # Sort test set by 'id'
    df_test = df_test.sort_values(by=['id'])
    
    # Remove Duplicates for Test DF if necessary
    if remove_test_duplicates:
        print("Removing Test Duplicates")
        df_test = df_test.drop_duplicates(subset=['id'])
        df_test.reset_index(drop=True, inplace=True)
        print(df_test.shape)
        print(df_test.head())
    
    # Convert to Dataset
    dataset_test = datasets.Dataset.from_pandas(df_test[['id','question','response']].copy())
    
    # Create DatasetDict with only 'test'
    ds = datasets.DatasetDict({"test": dataset_test})
    
    print(ds)
    return ds

# %%
ds = prepare_test_data()

# %%
ds['test'][0]

# %%
# Load the dataset from the saved disk location
loaded_dataset_dict = datasets.DatasetDict.load_from_disk(ASKER_PREDS)

# Convert the 'data' part of the DatasetDict to a Pandas DataFrame
asker_df = loaded_dataset_dict['data'].to_pandas()

# Now 'loaded_df' contains your DataFrame
asker_df.head()

# %% [markdown]
# ### VLLM

# %%
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# %%
def extract_ans(step):
    pattern = r'<<.*?=(\d+)>>'
    match = re.search(pattern, step)
    
    if match:
        return float(match.group(1))
    
    # Fallback to original pattern
    number_pattern = r'[$]?[-+]?\d+(?:\.\d+)?(?:,\d+)*[$]?'
    matches = re.findall(number_pattern, step)
    
    if matches:
        cleaned_value = float(matches[-1].replace(",", "").replace(" ", "").replace("\n", "").replace("$", "").replace("x", ""))
        return cleaned_value
    
    return None

# %%
def split_into_steps(text):
    # Split the text by the newline character
    steps = text.split('\n')
    
    # Remove any empty strings and filter out steps that start with '####'
    steps = [step for step in steps if step.strip() and not step.strip().startswith('####')]
    
    return steps

# %%
questions = []
gt_answers = []
final_gt_answers = []
for item in ds['test']:
    questions.append(item['question'])
    gt_answers.append(item['response'])
    final_gt_answers.append(item['response'].split("####")[1].strip())

# %%
asker_first_steps = asker_df['subanswer']
asker_first_ans = asker_df['only_ans']

# %%
df_cot = pd.DataFrame({
    'question': questions,
    # 'answer': answers
    'asker_first_step': asker_first_steps,
    'asker_first_ans':asker_first_ans,
    'gt_final_ans': final_gt_answers
})

df_cot.head()

# %%
def construct_prompt(checkpoint_path, questions):
    """
    Formats a batch of questions and first steps using ChatML template.

    Args:
        checkpoint_path (str): Path to the checkpoint of the model.
        questions (list of str): List of math word problems.
        first_steps (list of str): List of first steps for each question.

    Returns:
        list of torch.Tensor: List of formatted and tokenized prompts ready for input.
    """
    tokenizer = get_chat_template(
        AutoTokenizer.from_pretrained(checkpoint_path),  # Adjust this function to your needs
        chat_template="chatml",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gemma"},
        map_eos_token=True
    )

    formatted_questions = []
    input_token_counts = []

    for question in questions:
        formatted_question = tokenizer.apply_chat_template(
            [{"from": "human", "value": f"### Instruction:\nSolve the following Math Word Problem\n\n### Input:\n{question}"}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        
        # Calculate number of input tokens
        input_token_count = len(formatted_question.squeeze().tolist())
        input_token_counts.append(input_token_count)
    
        formatted_questions.append(formatted_question)
    
    return formatted_questions, tokenizer

# %%
def format_prompts_batch(checkpoint_path, questions, first_steps, flag):
    """
    Formats a batch of questions and first steps using ChatML template.

    Args:
        checkpoint_path (str): Path to the checkpoint of the model.
        questions (list of str): List of math word problems.
        first_steps (list of str): List of first steps for each question.

    Returns:
        list of torch.Tensor: List of formatted and tokenized prompts ready for input.
    """
    tokenizer = get_chat_template(
        AutoTokenizer.from_pretrained(checkpoint_path),  # Adjust this function to your needs
        chat_template="chatml",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gemma"},
        map_eos_token=True
    )

    formatted_questions = []
    input_token_counts = []

    for question, first_step in zip(questions, first_steps):
        if flag:
            formatted_question = tokenizer.apply_chat_template(
                [{"from": "human", "value": f"### Instruction:\nCalculate only the first step for the following Math Word Problem\n\n### Input:\n{question}"}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")
        else:
            formatted_question = tokenizer.apply_chat_template(
                [
                    {"from": "human", "value": f"### Instruction:\nCalculate only the first step for the following Math Word Problem\n\n### Input:\n{question}"},
                    {"from": "gemma", "value": f"{first_step}"},
                    {"from": "human", "value": f"### Instruction:\nContinue generating the entire answer from the next step\n\n"}
                ],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")
        
        # Calculate number of input tokens
        input_token_count = len(formatted_question.squeeze().tolist())
        input_token_counts.append(input_token_count)
    
        formatted_questions.append(formatted_question)
    
    return formatted_questions, tokenizer

# %%
def generate_steps_batch(checkpoint_path, max_seq_len, formatted_questions, tokenizer, cost_per_token):
    """
    Generates steps for a batch of math word problems and calculates the token cost.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        max_seq_len (int): Maximum sequence length for token generation.
        formatted_questions (list of torch.Tensor): List of tokenized prompts for the batch.
        tokenizer (AutoTokenizer): The tokenizer used for decoding.
        input_token_counts (list of int): List of input token counts for each question.
        cost_per_token (float): The cost per token for the model.

    Returns:
        list of str: List of generated texts for the batch.
        float: Total cost for all tokens processed.
    """
    llm = LLM(model=checkpoint_path)
    sampling_params = SamplingParams(temperature=0, max_tokens=max_seq_len, stop=["### Instruction", "### Input"])
    
    predictions = []
    total_tokens_used = 0  # To keep track of total tokens (input + output)
    total_time = 0
    for formatted_question in formatted_questions:
        token_ids = formatted_question.squeeze().tolist()
        decoded_question = tokenizer.decode(token_ids, skip_special_tokens=True)
        start_time = time.time()
        # Generate outputs for each question
        outputs = llm.generate(decoded_question, sampling_params)
        end_time = time.time()
        total_time += end_time - start_time
        for output in outputs:   
            # total_time += (output.metrics.finished_time - output.metrics.arrival_time)
            generated_text = output.outputs[0].text
            predictions.append(generated_text)
            
            # Calculate output token count
            output_token_count = len(tokenizer.encode(generated_text))
            
            # Add input + output tokens to total
            total_tokens_used += output_token_count
    
    # Calculate the total cost
    total_cost = total_tokens_used * cost_per_token

    # Clean up to free memory
    torch.cuda.empty_cache()
    del llm
    del sampling_params
    gc.collect()

    return predictions, total_cost, total_time

# # %%
# prompts, tokenizer = construct_prompt(checkpoint_COT_all, questions)
# predictions, cost_all, time_all = generate_steps_batch(checkpoint_COT_all, MAX_SEQ_LENGTH, prompts, tokenizer, 1.0)
# df_all = pd.DataFrame({
#     'question': questions,
#     'entire_COT': predictions,
#     'gt_final_ans': final_gt_answers
# })
# df_all.head()
# # print(f"time_all={time_all}")
# # time_all.to_pickle("/cluster/project/sachan/piyushi/results/qwen/0.5_1.5/time_all.pkl")

# # %%
# # df_all.to_pickle("/cluster/project/sachan/piyushi/results/qwen/0.5_1.5/dfall.pkl")

# # %%
# flag = True

# first_steps = [""] * len(questions)
# first_step_prompts, tokenizer = format_prompts_batch(checkpoint_path, questions, first_steps, flag)
# first_step_predictions, cost_first_step, time_FS = generate_steps_batch(
#     checkpoint_path, MAX_SEQ_LENGTH, first_step_prompts, tokenizer, 1.0)
# df_cot['pred_first_step'] = first_step_predictions
# df_cot.head()
# print(f"time_FS={time_FS}")
# time_FS.to_pickle("/cluster/project/sachan/piyushi/results/qwen/0.5_1.5/time_FS.pkl")


# # %%
# df_cot.to_pickle("/cluster/project/sachan/piyushi/results/qwen/0.5_1.5/dfcot.pkl")

# %%
# df_all = pd.read_pickle("/cluster/project/sachan/piyushi/results/qwen/0.5_1.5/dfall.pkl")

# # %%
df_cot = pd.read_pickle('/cluster/project/sachan/piyushi/results/qwen/0.5_1.5/dfcot.pkl')

# # %%
# cost_replaced = acc_replaced = 0

# %%
true_rows = []
time_replacedFS = 0

for idx, first_step in enumerate(df_cot['pred_first_step']):
    ans = extract_ans(first_step)
    ans_right = df_cot.loc[idx, 'asker_first_ans']
    if ans == ans_right:
        correct_step = first_step
        # true_rows.append({
        # 'question': df_cot.loc[idx, 'question'],
        # # 'true_answer': ds['test'][idx]['response'],
        # 'pred_first_step': first_step,
        # 'gt_final_ans': df_cot.loc[idx, 'gt_final_ans']
        # })
    else:
        correct_step = df_cot.loc[idx, 'asker_first_step'] 
    
    true_rows.append({
        'question': df_cot.loc[idx, 'question'],
        # 'true_answer': ds['test'][idx]['response'],
        'pred_first_step': correct_step,
        'gt_final_ans': df_cot.loc[idx, 'gt_final_ans']
    })

# Create a new DataFrame with only the rows where the condition was true
df_correct = pd.DataFrame(true_rows)
df_correct.head()

# %%
flag = False
remaining_step_prompts, tokenizer = format_prompts_batch(
    checkpoint_path, df_correct['question'].tolist(), df_correct['pred_first_step'].tolist(), flag)
# remaining_step_predictions, cost_final, time_correctFS = generate_steps_batch(
#     checkpoint_path, MAX_SEQ_LENGTH, remaining_step_prompts, tokenizer, 1.0)
remaining_step_predictions, cost_replaced, time_replacedFS = generate_steps_batch(
    checkpoint_path, MAX_SEQ_LENGTH, remaining_step_prompts, tokenizer, 1.0)
df_correct['remaining_steps'] = remaining_step_predictions
df_correct.head()

# # %%
# def get_final_ans(step):
#     # Find all matches of the pattern <<...=value>> and extract the last one
#     pattern = r'<<.*?=(\d+(?:\.\d+)?)>>'
#     matches = re.findall(pattern, step)
    
#     if matches:
#         # Return the last match (the answer from the last step)
#         return float(matches[-1])
    
#     return None

# # %%
# count = 0
# total = 0

# for index, row in df_all.iterrows(): 
#     steps = row['entire_COT']
#     total += 1
#     if "####" in steps:
#         pred_final = steps.split("####")[1].strip()
#     else:
#         pred_final = get_final_ans(steps)
#     gt_final = df_all.loc[index, 'gt_final_ans']
#     if pred_final == gt_final: 
#         count += 1
# acc_all = count / total *100
# print(total, acc_all)

# #%%
# count = 0
# total = 0
# acc_socratic = 0
# for index, row in df_correct.iterrows(): 
#     steps = row['remaining_steps']
#     total += 1
#     if "## Final Answer: " in steps:
#         pred_final = steps.split("## Final Answer:")[1].strip()
#     else:
#         pred_final = get_final_ans(steps)
#     gt_final = df_correct.loc[index, 'gt_final_ans']
#     if pred_final == gt_final: 
#         count += 1
# acc_socratic = count / 1319 *100
# num_correct_FS = total
# print(total, acc_socratic)

# # %%
# count = 0
# total = 0
# acc_replaced = cost_final = 0
# for index, row in df_correct.iterrows(): 
#     steps = row['remaining_steps']
#     total += 1
#     if "## Final Answer: " in steps:
#         pred_final = steps.split("## Final Answer:")[1].strip()
#     else:
#         pred_final = get_final_ans(steps)
#     gt_final = df_correct.loc[index, 'gt_final_ans']
#     if pred_final == gt_final: 
#         count += 1
# acc_replaced = count / 1319 *100
# print(total, acc_replaced)

# # %%
# cnt=0
# tot=0
# for idx, first_step in enumerate(df_cot['pred_first_step']):
#     tot+=1
#     ans = extract_ans(first_step)
#     ans_right = df_cot.loc[idx, 'asker_first_ans']
#     if ans == ans_right:
#         cnt += 1
# acc_first = cnt/tot *100
# print(tot, acc_first)

# # %%
# def calculate_total_cost(df_all, tokenizer, cost_per_token, flag):
#     total_tokens_used = 0

#     # Tokenize and count tokens for each prediction in the 'entire_COT' column
#     if flag:
#         for prediction in df_all['entire_COT']:
#             token_ids = tokenizer.encode(prediction, add_special_tokens=False)
#             total_tokens_used += len(token_ids)
#     else:
#         for prediction in df_all['pred_first_step']:
#             token_ids = tokenizer.encode(prediction, add_special_tokens=False)
#             total_tokens_used += len(token_ids)
    
#     # Calculate the total cost
#     total_cost = total_tokens_used * cost_per_token
#     return total_cost

# cost_all = calculate_total_cost(df_all, tokenizer, 1.0, True)
# cost_first_step = calculate_total_cost(df_cot, tokenizer, 1.0, False)

# # %%
# data = {
#     'Metric': ['cost_first_step', 'cost_final_only', 'cost_replaced', 'cost_all', 'total_cost_replaced', 'total_cost_FS_correct', 'len(df_cot)', 'len(df_correct)', 'acc_all', 'acc_FS','acc_correct_FS','acc_FS_asker'],
#     'Value': [cost_first_step, cost_final, cost_replaced, cost_all, cost_first_step+cost_replaced, cost_first_step+cost_final, len(df_cot), num_correct_FS, acc_all, acc_first, acc_socratic, acc_replaced]
# }

# data = {
#     'Metric': ['cost_replaced', 'acc_FS_asker'],
#     'Value': [cost_replaced,  acc_replaced]
# }

# time_data = {
#     'Times': ['time_all', 'time_FS', 'time_correctFS', 'time_replacedFS'],
#     'Value': [time_all,  time_FS, time_correctFS, time_replacedFS]
# }
# df_times = pd.DataFrame(time_data)
df_times = pd.read_pickle("/cluster/project/sachan/piyushi/results/qwen/0.5_1.5/df_times.pkl")
df_times.loc[3,"Value"]=time_replacedFS
df_times
# df_results.to_pickle("/cluster/project/sachan/piyushi/results/qwen/0.5_3/dfresults.pkl")

# %%
# df_results.loc[4,"Value"]=224042.0
# df_results.loc[2,"Value"]=135546.0
# df_results.loc[11,"Value"]=0.534496

# # %%
# df_results.to_pickle("/cluster/project/sachan/piyushi/results/qwen/0.5_1.5/dfresults2.pkl")

# %%
df_times.to_pickle("/cluster/project/sachan/piyushi/results/qwen/0.5_1.5/df_times.pkl")

# %%
# df_results=pd.read_pickle("/cluster/project/sachan/piyushi/results/qwen/0.5_0.5/dfresults2.pkl")

# %%
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--output_path', required=True, help="Path for output")
#     parser.add_argument('--checkpoint_path', required=True, help="Path to the model checkpoint")
#     parser.add_argument('--checkpoint_cot_all', required=True, help="Path to COT checkpoint")
#     parser.add_argument('--asker_preds', required=True, help="Path for asker predictions")
#     args = parser.parse_args()

#     # Use the arguments in the script
#     OUTPUT_PATH = args.output_path
#     checkpoint_path = args.checkpoint_path
#     checkpoint_COT_all = args.checkpoint_cot_all
#     ASKER_PREDS = args.asker_preds

# # %%
# df_results

# # %%
# df_cot.head()

# # %%
# df_all.head()

# # %%
# def first_and_final(index, text):
#     first=final=""
#     first = text.split("\n")[0].strip()
#     if "####" in text:
#         final = text.split("####")[1].strip()
#     else: print(index, text)
#     return first, final

# count_first = 0
# tot_first = 0
# count_final = 0
# tot_final = 0

# for index, row in df_cot.iterrows(): 
#     ans = row['pred_first_step']
#     if "### Answer:\n" in ans:
#         tot_first += 1
#         only_ans = ans.split("### Answer:\n")[1].strip()
#         pred_first, pred_final = first_and_final(index, only_ans)
#         gt_ans = row['answer']
#         gt_first, gt_final = first_and_final(index, gt_ans)
#         pred_first_ans = get_final_ans(pred_first)
#         gt_first_ans = get_final_ans(gt_first)
#         if gt_first_ans == pred_first_ans:
#             count_first += 1
#             if pred_final != "":
#                 tot_final += 1
#                 if gt_final == pred_final:
#                     count_final += 1
#         # gt_ans = row['answer']
#     # gt_first, gt_final = first_and_final(index, gt_ans)
#     # pred_first_ans = get_final_ans(pred_first)
#     # gt_first_ans = get_final_ans(gt_first)
#     # if gt_first_ans == pred_first_ans:
#     #     count_first += 1
#     # if pred_final != "" and gt_final == pred_final:
#     #     count_final += 1
# print(tot_first, tot_final)
# print(f"First answer accuracy = {count_first/tot_first}")
# print(f"Final answer accuracy = {count_final/tot_final}")


