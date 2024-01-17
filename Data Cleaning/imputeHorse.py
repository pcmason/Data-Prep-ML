'''
Uses multiple different statistics for data imputation on the Horse Colic dataset
that has 26 input vars and determines if a horse lives (1) or dies (2) from Colic
'''
import pandas as pd
import warnings
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Load Horse Colic dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv'
df = pd.read_csv(url, header=None, na_values='?')
# Summarize first few rows
#print(df.head())
# Summarize number of rows with missing values in each column
print('Go through every row and output # of missing values:')
for i in range(df.shape[1]):
    # Count number of rows with missing values
    n_miss = df[[i]].isnull().sum()
    perc = n_miss / df.shape[0] * 100
    print('\t>%d, Missing: %d (%.1f%%)' % (i, n_miss, perc))

# Split data into input & output
data = df.values
# How to separate when output is not at the end of the sheet
ix = [i for i in range(data.shape[1]) if i != 23]
x, y = data[:, ix], data[:, 23]
# Evaluate each strategy on the dataset
results = list()
strats = ['mean', 'median', 'most_frequent', 'constant']
# Create modeling pipeline [this is to make a prediction on]
mean_pipe = Pipeline(steps=[('i', SimpleImputer(strategy='mean')), ('m', RandomForestClassifier())])
print('\nTry mean, median, most_frequent & a constant for data imputation:')
for s in strats:
    # Create modeling pipeline
    pipeline = Pipeline(steps=[('i', SimpleImputer(strategy=s)), ('m', RandomForestClassifier())])
    # Evaluate model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # Store results
    results.append(scores)
    print('\t>%s %.3f (%.3f)' % (s, np.mean(scores), np.std(scores)))

# Plot model performance for comparison
plt.boxplot(results, labels=strats, showmeans=True)
plt.show()

# Fit the model
mean_pipe.fit(x, y)
# Define new data
row = [2, 1, 530101, 38.50, 66, 28, 3, 3, np.nan, 2, 5, 4, 4, np.nan, np.nan, np.nan, 3, 5, 45.00, 8.40, np.nan, np.nan, 2, 11300, 00000, 00000, 2]
# Make a prediction
yhat = mean_pipe.predict([row])
# Summarize prediction
print('\nPredicted Class Given Input Data: %d' % yhat[0])
