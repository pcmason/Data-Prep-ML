'''
Worked example code transforming the categorical input variables of the Breast Cancer dataset using
One-Hot Encoding & Ordinal Encoding methods from the sklearn libary.
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score

# Load in the data
dataset = pd.read_csv('breast-cancer.csv', header=None)
# Retrieve array of data
data = dataset.values
# Separate input & output columns
x = data[:, :-1].astype('str')
y = data[:, -1].astype('str')
# Split dataset into train & test splits
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
# Create list for the 2 transformation strategies used
strats = [OrdinalEncoder(), OneHotEncoder()]
for s in strats:
    # Encode input variables
    encoder = s
    encoder.fit(x_train)
    x_train = encoder.transform(x_train)
    x_test = encoder.transform(x_test)
    # Ordinal encode target variable
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)
    # Define the model
    model = LogisticRegression()
    # Fit on training data
    model.fit(x_train, y_train)
    # Predict on test set
    yhat = model.predict(x_test)
    # Evaluate predictions
    accuracy = accuracy_score(y_test, yhat)
    print('%s Accuracy: %.3f' % (s, accuracy * 100))


