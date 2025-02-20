import pandas as pd

# Data definition
data = [
    {'Model': '2', 'Cost': [178421.0] + [166980.0, 169028.0], 
     'Accuracy': [26.99] + [24.18, 28.05]},
    {'Model': '7', 'Cost': [175703.0] + [167993.0, 168749.0], 
     'Accuracy': [54.96] + [38.66, 48.36]},
    # {'Model': '3', 'Cost': [171215.0] + [200190.0, 197374.0, 197801.0, 199543.0], 
    #  'Accuracy': [64.06] + [50.34, 59.36, 60.65, 64.44]},
    # {'Model': '7', 'Cost': [171465.0] + [224042.0, 219908.0, 222334.0, 217131.0], 
    #  'Accuracy': [72.55] + [53.44, 62.39, 62.47, 66.33]},
]

# Create columns for feature sets
columns = ['All samples original', '2B', '7B']

# Prepare rows data with cost-accuracy pairs
rows = []
for entry in data:
    # Combine cost and accuracy into pairs
    pairs = list(zip(entry['Cost'], entry['Accuracy']))
    rows.append([entry['Model']] + pairs)

multi_columns = pd.MultiIndex.from_tuples([('Model', '')] + [('All samples original', '')] + [('Asker Sizes', col) for col in columns[1:]])
# Create the final DataFrame
df = pd.DataFrame(rows, columns=multi_columns)
# Save the DataFrame to a CSV file
df.to_csv('model_performance_pairs.csv', index=False)
print(df)