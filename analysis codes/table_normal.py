import pandas as pd

# Data definition
data = [
    {'Model': '2', 'Cost': [178421.0] + [104054.0, 100213.0], 
     'Accuracy': [26.99] + [18.27, 22.06]},
    {'Model': '7', 'Cost': [175703.0] + [105726.0, 118053.0], 
     'Accuracy': [54.96] + [28.88, 36.46]},
    # {'Model': '3', 'Cost': [171215.0] + [115232.0, 136336.0, 149441.0, 146033.0], 
    #  'Accuracy': [64.06] + [31.61, 41.93, 47.38, 44.80]},
    # {'Model': '7', 'Cost': [171465.0] + [139225.0, 157874.0, 166015.0, 172235.0], 
    #  'Accuracy': [72.55] + [31.46, 40.56, 43.13, 48.21]},
]

# Create columns for feature sets
columns = ['All samples', '2B', '7B']

# Prepare rows data with cost-accuracy pairs
rows = []
for entry in data:
    # Combine cost and accuracy into pairs
    pairs = list(zip(entry['Cost'], entry['Accuracy']))
    rows.append([entry['Model']] + pairs)

# Create the MultiIndex for the header
multi_columns = pd.MultiIndex.from_tuples([('Model', '')] + [('All samples', '')] + [('Asker Sizes', col) for col in columns[1:]])

# Create the final DataFrame
df = pd.DataFrame(rows, columns=multi_columns)
print(df)