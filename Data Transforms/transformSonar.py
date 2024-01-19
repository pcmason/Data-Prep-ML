'''
This file uses StandardScalar & MinMaxScalar to standardize & normalize the sonar dataset.
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Load sonar dataset
dataset = pd.read_csv('sonar.csv', header=None)
data = dataset.values
# Create a histogram of the unchanged variables
dataset.hist()
plt.show()
# Separate into input & output variables
x, y = data[:, :-1], data[:, -1]
# Ensure inputs are floats & output is an integer label
x = x.astype('float32')
y = LabelEncoder().fit_transform(y.astype('str'))
# Define & configure KNN model
model = KNeighborsClassifier()
# Evaluate model [baseline performance no rescaling of data]
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('\nBaseline Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

# Variable to combine normalization & standardization into a loop
strats = [MinMaxScaler(), StandardScaler()]
count = 0
for s in strats:
    trans = s
    # Define the pipeline
    pipeline = Pipeline(steps=[('t', trans), ('m', model)])
    # Evaluate the pipeline for normalization & standardization
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(pipeline, x, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # Special if statement to specify between 2 different outputs
    if count == 0:
        print('\nNormalized Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
        count += 1
    else:
        print('\nStandardized Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


# Now normalize the variables for the sonar dataset
# Retrieve just the numeric input variables
n_data = dataset.values[:, :-1]
# Perform a robust scalar transform of the dataset
n_trans = MinMaxScaler()
n_data = n_trans.fit_transform(n_data)
# Convert array to DF
n_dataset = pd.DataFrame(n_data)
# Output histograms of normalized variables
n_dataset.hist()
plt.show()

# Standardize variables for sonar dataset
s_data = dataset.values[:, :-1]
# Perform a robust scalar transform of the dataset
s_trans = StandardScaler()
s_data = s_trans.fit_transform(s_data)
# Convert back to dataframe and output as histogram
s_dataset = pd.DataFrame(s_data)
s_dataset.hist()
plt.show()