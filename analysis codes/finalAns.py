import json
import matplotlib.pyplot as plt
import re

# Function to load JSONL file
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data

# Load data from both files
predictions_path = "/Users/piyushigoyal/Documents/ETH-Zurich/LRE/Shridhar/GSM8K/test/llama-7B_COT/predictions.jsonl"
test_socratic_path = "/Users/piyushigoyal/Documents/ETH-Zurich/LRE/Shridhar/GSM8K/test/test_socratic.jsonl"

predictions_data = load_jsonl(predictions_path)
test_socratic_data = load_jsonl(test_socratic_path)


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
    answers = []
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
                answers.append(ans)
    return answers

def extract_final_ans(answer_text):
    """Extracts the final gt answer"""
    parts = answer_text.split('####')
    pattern = r'[$]?[-+]?\d+(?:\.\d+)?(?:,\d+)*[$]?'
    match = re.findall(pattern, parts[1])
    return float(match[-1].replace(",", "").replace(" ", "").replace("\n", "").replace("$", "").replace("x", ""))

def extract_pred_ans(completion_text):
    """Extracts the final prediction answer"""
    parts = completion_text.split("The answer is ")
    pattern = r'[$]?[-+]?\d+(?:\.\d+)?(?:,\d+)*[$]?'
    match = re.findall(pattern, parts[1])
    return float(match[-1].replace(",", "").replace(" ", "").replace("\n", "").replace("$", "").replace("x", ""))

def ans_similarity(num1:list, num2):
    """Computes the semantic similarity between two pieces of text using spaCy."""
    if num2 in num1:
        return 1.0
    return 0.0

# Sample data to test functions
assert len(predictions_data) == len(test_socratic_data), "Data length mismatch."


# sample_socratic_answer = test_socratic_data[0]['answer']
# sample_predictions_completion = predictions_data[0]['completion']

# first_subquestion_answer = extract_first_subquestion_answer(sample_socratic_answer)

# socratic_final_ans = extract_final_ans(sample_socratic_answer)
# print(f"socratic_final_ans = {socratic_final_ans}")

# predictions_final_ans = extract_pred_ans(sample_predictions_completion)
# print(f"predictions_final_ans = {predictions_final_ans}")

# first_step = extract_first_step(sample_predictions_completion)

# first_ans_similarity = ans_similarity(first_step, first_subquestion_answer)
# final_ans_similarity = ans_similarity([socratic_final_ans], predictions_final_ans)
# print(f"first_subquestion_answer = {first_subquestion_answer}")
# print(f"first_step_COT = {first_step}")
# print(f"first answer similarity = {first_ans_similarity}")
# print(f"final answer similarity = {final_ans_similarity}")

# Looping through the datasets
results = []
count_1 = 0
count_0 = 0
count_base = 0

for prediction, socratic in zip(predictions_data, test_socratic_data):
    first_subquestion_answer = extract_first_subquestion_answer(socratic['answer'])
    first_step = extract_first_step(prediction['completion'])

    socratic_final_ans = extract_final_ans(socratic['answer'])
    predictions_final_ans = extract_pred_ans(prediction['completion'])

    first_ans_similarity = ans_similarity(first_step, first_subquestion_answer)
    final_ans_similarity = ans_similarity([socratic_final_ans], predictions_final_ans)
    if first_ans_similarity == 1.0 and final_ans_similarity == 1.0:
        count_1 += 1
    elif final_ans_similarity == 1.0:
        count_base += 1
    # elif first_ans_similarity == 1.0 and final_ans_similarity == 0.0:
    #     count_0 += 1
    
    # Storing results
    results.append({
        "question_id": prediction.get('id', socratic.get('id')),  # Assuming 'id' is present and consistent
        "first_subquestion_answer": first_subquestion_answer,
        "first_step_of_COT": first_step,
        "gt_answer": socratic_final_ans,
        "final_COT_answer": predictions_final_ans,
        "final_ans_similarity": final_ans_similarity
    })

# for prediction, socratic in zip(predictions_data, test_socratic_data):
#     first_subquestion_answer = extract_first_subquestion_answer(socratic['answer'])
#     first_step = extract_first_step(prediction['completion'])

#     socratic_final_ans = extract_final_ans(socratic['answer'])
#     predictions_final_ans = extract_pred_ans(prediction['completion'])

#     first_ans_similarity = ans_similarity(first_step, first_subquestion_answer)
#     final_ans_similarity = ans_similarity([socratic_final_ans], predictions_final_ans)
    
#     # Storing results
#     results.append({
#         "question_id": prediction.get('id', socratic.get('id')),  # Assuming 'id' is present and consistent
#         "gt_answer": socratic_final_ans,
#         "COT_answer": predictions_final_ans,
#         "similarity_score": final_ans_similarity
#     })
# # Outputting the first few results to check
output_file_path = "/Users/piyushigoyal/Documents/ETH-Zurich/LRE/Shridhar/GSM8K/test/outputs/baselines/output_llama-7B_results.json"
with open(output_file_path, 'w') as outfile:
    json.dump(results, outfile, indent=4)

scores = ['Baseline', 'FS']
counts = [count_base, count_1]
print(count_base,count_1)
# Creating the bar chart
plt.figure(figsize=(8, 5))
plt.bar(scores, counts, color=['green', 'red'])
plt.xlabel('Similarity Scores')
plt.ylabel('Number of Questions')
plt.title('Number of Questions with Similarity Scores of 1 and 0')
plt.show()

# print("Results have been successfully exported to", output_file_path)

# prediction_files = [
#     "/Users/piyushigoyal/Documents/ETH Zürich/Mrinmaya's lab/Shridhar/GSM8K/test/llama-7B_COT/predictions.jsonl",
#     "/Users/piyushigoyal/Documents/ETH Zürich/Mrinmaya's lab/Shridhar/GSM8K/test/llama-13B_COT/predictions.jsonl",
#     "/Users/piyushigoyal/Documents/ETH Zürich/Mrinmaya's lab/Shridhar/GSM8K/test/llama-70B_COT/predictions.jsonl",
#     "/Users/piyushigoyal/Documents/ETH Zürich/Mrinmaya's lab/Shridhar/GSM8K/test/mistral-7B_COT/predictions.jsonl",
#     "/Users/piyushigoyal/Documents/ETH Zürich/Mrinmaya's lab/Shridhar/GSM8K/test/mistral-7B-instruct_COT/predictions.jsonl"
# ]
# output_base_dir = "/Users/piyushigoyal/Documents/ETH Zürich/Mrinmaya's lab/Shridhar/GSM8K/test/outputs"