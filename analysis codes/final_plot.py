import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Creating the dataframe with the provided values
data = {
    'Cost': [176567.0, 100213.0],
    'Accuracy': [27, 17.89],
    'Label': ['COT_ALL', 'FS+Asker']
}

df = pd.DataFrame(data)

# Plotting the graph
plt.figure(figsize=(8, 6))
plt.scatter(df['Cost'], df['Accuracy'], color='blue')

# Adding labels to the points
for i, label in enumerate(df['Label']):
    plt.text(df['Cost'][i] - 1000, df['Accuracy'][i] - 0.1, label, fontsize=9, ha='right')

# Setting titles and labels
plt.title('Accuracy vs Cost')
plt.xlabel('Cost')
plt.ylabel('Accuracy (%)')

# Show the plot
plt.grid(True)
plt.show()

# # Display the dataframe to user
# import ace_tools as tools; 
# tools.display_dataframe_to_user(name="Accuracy vs Cost Data", dataframe=df)