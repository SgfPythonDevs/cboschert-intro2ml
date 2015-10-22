__author__ = 'Chad'
import pandas as pd
import numpy as np

# Pandas has its own read_csv method; it's smart enough to infer data types
train_df = pd.read_csv('.\\data\\train.csv', header=0)

# Pandas introduces two data structures: Series and DataFrames
# Series are one-dimensional similar to arrays
s = pd.Series([123, 'Chad', 3.14, 'Another String'])
s2 = pd.Series([123, 'Chad', 3.14, 'Another String'], index=['Id', 'Name', 'Value', 'Comment'])
s3 = train_df['Name']

# DataFrame is a tabular structure like a spreadsheet; it's a collection of Series that share the same index
df = pd.DataFrame({
    'PersonId': [1, 2, 3, 4, 5],
    'Name': ['Chad', 'Fred', 'Mark', 'Wesley', 'Ben']
}, columns=['PersonId', 'Name'])