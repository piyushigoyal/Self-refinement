import json
import csv

# Define the input JSONL file and output CSV file
jsonl_file = 'train_socratic.jsonl'
csv_file = 'socratic_train.csv'

# Open the JSONL file and CSV file
with open(jsonl_file, 'r') as infile, open(csv_file, 'w', newline='') as outfile:
    # Define the CSV writer
    writer = csv.writer(outfile)
    
    # Write the header row to the CSV
    writer.writerow(['id', 'question', 'answer'])
    
    # Iterate through each line in the JSONL file
    for idx, line in enumerate(infile):
        # Parse the JSONL line to a Python dictionary
        data = json.loads(line)
        
        # Extract the question and answer fields
        question = data.get('question', '')
        answer = data.get('answer', '')
        
        # Write the data to the CSV file with an ID
        writer.writerow([idx + 1, question, answer])

print(f"Conversion complete! The CSV file is saved as {csv_file}.")