import pandas as pd
import numpy as np
import datasets
ASKER_PREDS = "/cluster/project/sachan/piyushi/asker_predictions/qwen_7/"

# Load the dataset from the saved disk location
loaded_dataset_dict1 = datasets.DatasetDict.load_from_disk("/cluster/project/sachan/piyushi/asker_predictions/qwen_3/")
loaded_dataset_dict2 = datasets.DatasetDict.load_from_disk("/cluster/project/sachan/piyushi/asker_predictions/qwen_1.5/")

# Convert the 'data' part of the DatasetDict to a Pandas DataFrame
df1 = loaded_dataset_dict1['data'].to_pandas()
df2 = loaded_dataset_dict2['data'].to_pandas()

# Now 'loaded_df' contains your DataFrame
# print(df1.equals(df2))
print(df2.head(10))