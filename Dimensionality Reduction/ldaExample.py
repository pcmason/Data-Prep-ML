'''
Program that creates a multi-class classification dataset with 1000 examples, 20 features & only 15 of the features are
relevant with 10 classes as the output. Use the sklearn Linear Discriminant Analysis for dimensionality reduction.
'''

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


# Method to create classification dataset
def get_dataset():
    x, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7, n_classes=10)
    return x, y


# Method that creates list of all 20 models to evaluate
def get_models():
    models = dict()
    for i in range(1, 10):
        # Create pipeline to do LDA then run the LDA output on the model
        steps = [('lda', LinearDiscriminantAnalysis(n_components=i)), ('m', GaussianNB())]
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
plt.show()

# Now define the model using 9 features based on results above
steps = [('lda', LinearDiscriminantAnalysis(n_components=9)), ('m', GaussianNB())]
model = Pipeline(steps=steps)
# Fit model on whole dataset
model.fit(x, y)
# Make a single prediction on new row of data
row = [[2.3548775,-1.69674567,1.6193882,-1.19668862,-2.85422348,-2.00998376,16.56128782,2.57257575,9.93779782,0.43415008,6.08274911,2.12689336,1.70100279,3.32160983,13.02048541,-3.05034488,2.06346747,-3.33390362,2.45147541,-1.23455205]]
yhat = model.predict(row)
print('\nPredicted Class: %d' % yhat[0])