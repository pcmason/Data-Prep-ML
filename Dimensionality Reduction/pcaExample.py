'''
Program that creates a binary classification dataset with 1000 examples, 20 features & only 15 of the features are
relevant. Use the sklearn Principal Component Analysis for dimensionality reduction on the dataset.
'''

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# Method to create classification dataset
def get_dataset():
    x, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
    return x, y


# Method that creates list of all 20 models to evaluate
def get_models():
    models = dict()
    for i in range(1, 21):
        # Create pipeline to do PCA then run the PCA output on the model
        steps = [('pca', PCA(n_components=i)), ('m', LogisticRegression())]
        models[str(i)] = Pipeline(steps=steps)
    return models


# Evaluate a given model using cross-validation
def evaluate_model(model, x, y):
    # Use 10 fold repeating CV with 3 repeats
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# Define dataset
x, y = get_dataset()
# Get the models for evaluation
models = get_models()
# Evaluate models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, x, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
# Plot model performance for visual comparison
plt.boxplot(results, labels=names, showmeans=True)
plt.xticks(rotation=45)
plt.show()

# Now define the model using 15 features based on results above
steps = [('pca', PCA(n_components=15)), ('m', LogisticRegression())]
model = Pipeline(steps=steps)
# Fit model on whole dataset
model.fit(x, y)
# Make a single prediction on new row of data
row = [[0.2929949,-4.21223056,-1.288332,-2.17849815,-0.64527665,2.58097719,0.28422388,-7.1827928,-1.91211104,2.73729512,0.81395695,3.96973717,-2.66939799,3.34692332,4.19791821,0.99990998,-0.30201875,-4.43170633,-2.82646737,0.44916808]]
yhat = model.predict(row)
print('\nPredicted Class: %d' % yhat[0])
