'''
Example code that uses the power transform methods Box-Cox & Yeo-Johnson to make the data from the Sonar dataset
have a more Gaussian (normal) distribution.
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, PowerTransformer, MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

# This is the method used to transform the data using either box-cox or yeo-johnson, function in action below
def powerTransform(scale, method, x, y):
    # Define pipeline
    scaler = scale
    power = PowerTransformer(method=method)
    model = KNeighborsClassifier()
    pipeline = Pipeline(steps=[('s', scaler), ('p', power), ('m', model)])
    # Evaluate pipeline
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(pipeline, x, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # Report performance
    print('%s Accuracy: %.3f (%.3f)' % (method, np.mean(n_scores), np.std(n_scores)))

# Load sonar dataset
dataset = pd.read_csv('sonar.csv', header=None)
data = dataset.values
# Separate into input & output variables
x, y = data[:, :-1], data[:, -1]
# Ensure inputs are floats & outputs are integer labels
x = x.astype('float32')
y = LabelEncoder().fit_transform(y.astype('str'))
# Define & configure KNN
model = KNeighborsClassifier()
# Evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# Report performance
print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

# Now use the box-cox method on the data
powerTransform(MinMaxScaler(feature_range=(1, 2)), 'box-cox', x, y)
# Now use the Yeo-Johnson method on the data
powerTransform(StandardScaler(), 'yeo-johnson', x, y)
