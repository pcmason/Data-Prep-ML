'''
This program identifies columns in the Oil Spill dataset with 1 or very few unique values in the column and removes those
columns. Also identifies duplicate rows in the Iris Flower dataset and removes those rows.
'''

from urllib.request import urlopen
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from matplotlib import pyplot

# URL of dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/oil-spill.csv'
# Load dataset
data = np.loadtxt(urlopen(path), delimiter=',')
# Summarize the number of unique values in each column
print('Percentage of unique variables per column:')
for i in range(data.shape[1]):
    num = len(np.unique(data[:, i]))
    # Get the percentage of unique values out of all the rows in the dataset
    percentage = float(num) / data.shape[0] * 100
    print('%d, %d, %.1f%%' % (i, num, percentage))

# This can also be done using the pandas nunique method
df = pd.read_csv('oil-spill.csv', header=None)
print('\nPandas version:\n', df.nunique())

# Show how many columns there are before deletion
print('\nShape of original dataset: ', df.shape)
# Get number of unique values for eacch column
counts = df.nunique()
# Record which columns should be deleted
to_del = [i for i, v in enumerate(counts) if v == 1]
print('\nColumn to delete: ', to_del)
# Drop the useless columns
new_df = df.drop(to_del, axis=1)
print('\nNew shape of dataset: ', new_df.shape)

# Delete values with 1% and less of their rows being unique
to_del = [i for i, v in enumerate(counts) if (float(v)/df.shape[0]*100) < 1]
print('\nRows with < 1% unique values: ', to_del)
# Drop useless columns
new_df = df.drop(to_del, axis=1)
print('\nNew shape of dataset: ', new_df.shape)

# Get the x and y values from the oil dataframe
x = data[:, :-1]
y = data[:, -1]
print('\nShape of x & y variables: ', x.shape, y.shape)
# Define the transform
transform = VarianceThreshold()
# Transform the input data
x_sel = transform.fit_transform(x)
print('\nNew shape: ', x_sel.shape)

# Define thresholds to check
thresholds = np.arange(0.0, 0.55, 0.05)
# Apply transform with eacch threshold
results = list()
print('\nRolling threshold test: ')
for t in thresholds:
    # Define the transform
    transform = VarianceThreshold(threshold=t)
    # Transform the input data
    x_sel = transform.fit_transform(x)
    # Determine the number of input features
    n_features = x_sel.shape[1]
    print('>Threshold=%.2f, Features=%d' % (t, n_features))
    # Store the result
    results.append(n_features)

# Plot the threshold vs the number of selected features
pyplot.plot(thresholds, results)
pyplot.show()

# Download the Iris Flowers dataset
iris_df = pd.read_csv('iris.csv', header=None)
# Calculate duplicates
dups = iris_df.duplicated()
# Report if there are any duplicates
print('\nAny duplicate rows?: ', dups.any())
# List all duplicate rows
print('\nList of duplicate rows:\n', iris_df[dups])

# Output shape of original dataset before row deletion
print('\nShape of original Iris dataset: ', iris_df.shape)
# Delete dupliccate rows
iris_df.drop_duplicates(inplace=True)
print('\nNew shape of Iris dataset: ', iris_df.shape)