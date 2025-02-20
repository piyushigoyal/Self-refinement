import os
import json
import matplotlib.pyplot as plt
import re

def load_jsonl(file_path):
    """Loads a JSONL file and returns a list of dictionaries."""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data

def extract_first_subquestion_answer(answer_text):
    """Extracts the final answer to the first subquestion after the equation."""
    # Split the answer into subquestions and answers
    parts = answer_text.split('**')
    answer_start=0
    if len(parts) > 1:
        first_answer = parts[1].split('##')[0].strip()
        if first_answer.lower().startswith(('define a variable', 'let\'s assume', 'let')):
            return 0.0
        # Regex to find text after the equation, accounting for optional <<>>
        match_patterns = (r'=.*?<<.*?>>', r"=\s+")
        if re.search(match_patterns[0], first_answer):
            match = re.search(match_patterns[0], first_answer)
            answer_start = match.end()
        elif re.search(match_patterns[1], first_answer):
            match = re.search(match_patterns[1], first_answer)
            answer_start = match.end()

        final_answer = first_answer[answer_start:].split('\n')[0].strip()
        # print(final_answer)
        pattern = r'[$]?[-+]?\d+(?:\.\d+)?(?:,\d+)*[$]?'
        matches = re.findall(pattern, final_answer)
        if  matches != []:
            ans = (float(matches[-1].replace(",", "").replace(" ", "").replace("\n", "").replace("$", "").replace("x", "")))
            # print(ans)
            return ans

    return 0.0

def extract_first_step(completion_text):
    """Extract the first sentence containing a mathematical equation from the completion."""
    # Split text into sentences
    # print(completion_text)
    sentences = completion_text.split('\n')
    # Regex to find sentences with equations (e.g., involving an equals sign)
    equation_regex = re.compile(r'(?:\$\s*)?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*[\+\-\*\/]\s*(?:\$\s*)?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*=\s*(?:\$\s*)?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*%)?')
    
    for sentence in sentences:
        if equation_regex.search(sentence):
            sentence_match = re.findall(equation_regex, sentence)
            # print(sentence_match)
            pattern = r'[$]?[-+]?\d+(?:\.\d+)?(?:,\d+)*[$]?'
            matches = re.findall(pattern, sentence_match[-1])
            # print(matches)
            if  matches != []:
                ans = (float(matches[-1].replace(",", "").replace(" ", "").replace("\n", "").replace("$", "").replace("x", "")))
                # print(ans)
                return ans
    return 0.0 

def extract_final_ans(answer_text):
    """Extracts the final gt answer"""
    parts = answer_text.split('####')
    pattern = r'[$]?[-+]?\d+(?:\.\d+)?(?:,\d+)*[$]?'
    if len(parts) > 1:
        match = re.findall(pattern, parts[1])
        if match != []:
            return float(match[-1].replace(",", "").replace(" ", "").replace("\n", "").replace("$", "").replace("x", ""))
    return 0.0

def extract_pred_ans(completion_text):
    """Extracts the final prediction answer"""
    parts = completion_text.split("The answer is ")
    pattern = r'[$]?[-+]?\d+(?:\.\d+)?(?:,\d+)*[$]?'
    if len(parts) > 1:
        match = re.findall(pattern, parts[1])
        if match != []:
            return float(match[-1].replace(",", "").replace(" ", "").replace("\n", "").replace("$", "").replace("x", ""))
    return 0.0

def ans_similarity(num1, num2):
    if num1 == num2:
        return 1.0
    return 0.0

def process_files(predictions_path, test_socratic_path, output_base_dir, index):
    test_socratic_data = load_jsonl(test_socratic_path)
    predictions_data = load_jsonl(predictions_path)

    results = []
    count_1 = 0
    count_0 = 0
    for prediction, socratic in zip(predictions_data, test_socratic_data):
        first_step = extract_first_step(prediction['completion'])
        if first_step == 0.0:
            continue
        first_subquestion_answer = extract_first_subquestion_answer(socratic['answer'])
        # socratic_final_ans = extract_final_ans(socratic['answer'])
        # predictions_final_ans = extract_pred_ans(prediction['completion'])

        first_ans_similarity = ans_similarity(first_step, first_subquestion_answer)
        if first_ans_similarity == 1.0:
            count_1 += 1
        else:
            count_0 += 1
        # final_ans_similarity = ans_similarity(socratic_final_ans, predictions_final_ans)
        # if first_ans_similarity == 1.0 and final_ans_similarity == 1.0:
        #     count_1 += 1
        # elif first_ans_similarity == 1.0 and final_ans_similarity == 0.0:
        #     count_0 += 1
        results.append({
            "question_id": prediction.get('id', socratic.get('id')),  # Assuming 'id' is present and consistent
            "first_subquestion_answer": first_subquestion_answer,
            "first_step_of_COT": first_step,
            # "gt_answer": socratic_final_ans,
            # "final_COT_answer": predictions_final_ans,
            "first_ans_similarity": first_ans_similarity
        })

    # Save results to JSON in a unique file for each model
    output_json_path = os.path.join(output_base_dir, f"output_{index}_results.jsonl")
    with open(output_json_path, 'w') as outfile:
        json.dump(results, outfile, indent=4)

    # Plotting and saving the histogram of similarity scores
    scores = ['1', '0']
    counts = [count_1, count_0]

    # Creating the bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(scores, counts, color=['green', 'red'])
    plt.xlabel('Similarity Scores')
    plt.ylabel('Number of Questions')
    plt.title('Number of Questions with Similarity Scores of 1 and 0')
    # plt.show()
    plt.grid(True)
    plot_path = os.path.join(output_base_dir, f"output_{index}_singleAns.png")
    plt.savefig(plot_path)
    plt.close()

# List of prediction files for different LLMs
prediction_files = [
    "/Users/piyushigoyal/Documents/ETH Zürich/Mrinmaya's lab/Shridhar/GSM8K/test/llama-7B_COT/predictions.jsonl",
    "/Users/piyushigoyal/Documents/ETH Zürich/Mrinmaya's lab/Shridhar/GSM8K/test/llama-13B_COT/predictions.jsonl",
    "/Users/piyushigoyal/Documents/ETH Zürich/Mrinmaya's lab/Shridhar/GSM8K/test/llama-70B_COT/predictions.jsonl",
    "/Users/piyushigoyal/Documents/ETH Zürich/Mrinmaya's lab/Shridhar/GSM8K/test/mistral-7B_COT/predictions.jsonl",
    "/Users/piyushigoyal/Documents/ETH Zürich/Mrinmaya's lab/Shridhar/GSM8K/test/mistral-7B-instruct_COT/predictions.jsonl"
]
test_socratic_path = "/Users/piyushigoyal/Documents/ETH Zürich/Mrinmaya's lab/Shridhar/GSM8K/test/test_socratic.jsonl"
output_base_dir = "/Users/piyushigoyal/Documents/ETH Zürich/Mrinmaya's lab/Shridhar/GSM8K/test/outputs_singleAns"

# Ensure the base output directory exists
os.makedirs(output_base_dir, exist_ok=True)

# Process each prediction file with a unique index
for index, file in enumerate(prediction_files, start=1):
    process_files(file, test_socratic_path, output_base_dir, index)
    print(f"Results for file {index} have been successfully exported to {output_base_dir}")
