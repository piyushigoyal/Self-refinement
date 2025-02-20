import torch
import pandas as pd
import numpy as np
import random
import datasets
from unsloth.chat_templates import get_chat_template
from vllm import LLM, SamplingParams
from transformers import AutoModel, AutoTokenizer


def calculate_total_cost(df_all, tokenizer, cost_per_token):
    total_tokens_used = 0

    for prediction in df_all['subanswer']:
            token_ids = tokenizer.encode(prediction, add_special_tokens=False)
            total_tokens_used += len(token_ids)
    
    # Calculate the total cost
    total_cost = total_tokens_used * cost_per_token
    return total_cost

ASKER_PREDS = "/cluster/project/sachan/piyushi/asker_predictions/qwen_7/"
loaded_dataset_dict = datasets.DatasetDict.load_from_disk(ASKER_PREDS)
asker_df = loaded_dataset_dict['data'].to_pandas()
checkpoint_path = "/cluster/project/sachan/piyushi/merged_models/qwen_7/checkpoint-236"
tokenizer = get_chat_template(
        AutoTokenizer.from_pretrained(checkpoint_path),  # Adjust this function to your needs
        chat_template="chatml",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gemma"},
        map_eos_token=True
    )
cost_all = calculate_total_cost(asker_df, tokenizer, 1.0)
print(cost_all)