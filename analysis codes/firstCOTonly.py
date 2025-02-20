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
predictions_path = "/Users/piyushigoyal/Documents/ETH Zürich/Mrinmaya's lab/Shridhar/GSM8K/test/mistral-7B_COT/predictions.jsonl"
test_socratic_path = "/Users/piyushigoyal/Documents/ETH Zürich/Mrinmaya's lab/Shridhar/GSM8K/test/test_socratic.jsonl"

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


def compare_texts(text1, text2):
    """A simple comparison function to check if text1 addresses text2."""
    # Convert texts to sets of words for simple overlap comparison
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    intersection = set1.intersection(set2)
    return len(intersection) / len(set1) if len(set1) > 0 else 0  # A basic measure of coverage

def ans_similarity(num1, num2):
    """Computes the semantic similarity between two pieces of text using spaCy."""
    if num1 == num2:
        return 1.0
    return 0.0

# Sample data to test functions
assert len(predictions_data) == len(test_socratic_data), "Data length mismatch."


# sample_socratic_answer = test_socratic_data[0]['answer']
# sample_predictions_completion = predictions_data[0]['completion']
# first_subquestion_answer = extract_first_subquestion_answer(sample_socratic_answer)
# first_step = extract_first_step(sample_predictions_completion)
# similarity_score = ans_similarity(first_step, first_subquestion_answer)
# print(f"first_subquestion_answer = {first_subquestion_answer}")
# print(f"first_step_COT = {first_step}")
# print(f"similarity = {similarity_score}")
# Looping through the datasets
results = []
similarity_scores = []
count_1 = 0
count_0 = 0

for prediction, socratic in zip(predictions_data, test_socratic_data):
    first_subquestion_answer = extract_first_subquestion_answer(socratic['answer'])
    first_step = extract_first_step(prediction['completion'])
    similarity_score = ans_similarity(first_step, first_subquestion_answer)
    if similarity_score == 1.0:
        count_1 += 1
    else:
        count_0 += 1
    # Storing results
    results.append({
        "question_id": prediction.get('id', socratic.get('id')),  # Assuming 'id' is present and consistent
        "first_subquestion_answer": first_subquestion_answer,
        "first_step_of_COT": first_step,
        "similarity_score": similarity_score
    })

# Outputting the first few results to check
output_file_path = "/Users/piyushigoyal/Documents/ETH Zürich/Mrinmaya's lab/Shridhar/GSM8K/test/output_results.json"
with open(output_file_path, 'w') as outfile:
    json.dump(results, outfile, indent=4)

scores = ['1', '0']
counts = [count_1, count_0]

# Creating the bar chart
plt.figure(figsize=(8, 5))
plt.bar(scores, counts, color=['green', 'red'])
plt.xlabel('Similarity Scores')
plt.ylabel('Number of Questions')
plt.title('Number of Questions with Similarity Scores of 1 and 0')
plt.show()

# print("Results have been successfully exported to", output_file_path)

prediction_files = [
    "/Users/piyushigoyal/Documents/ETH Zürich/Mrinmaya's lab/Shridhar/GSM8K/test/llama-7B_COT/predictions.jsonl",
    "/Users/piyushigoyal/Documents/ETH Zürich/Mrinmaya's lab/Shridhar/GSM8K/test/llama-13B_COT/predictions.jsonl",
    "/Users/piyushigoyal/Documents/ETH Zürich/Mrinmaya's lab/Shridhar/GSM8K/test/llama-70B_COT/predictions.jsonl",
    "/Users/piyushigoyal/Documents/ETH Zürich/Mrinmaya's lab/Shridhar/GSM8K/test/mistral-7B_COT/predictions.jsonl",
    "/Users/piyushigoyal/Documents/ETH Zürich/Mrinmaya's lab/Shridhar/GSM8K/test/mistral-7B-instruct_COT/predictions.jsonl"
]
test_socratic_path = "/Users/piyushigoyal/Documents/ETH Zürich/Mrinmaya's lab/Shridhar/GSM8K/test/test_socratic.jsonl"
# output_base_dir = "/Users/piyushigoyal/Documents/ETH Zürich/Mrinmaya's lab/Shridhar/GSM8K/test/outputs"