'''
Example code exploring the use of sklearn's Recursive Feature Elimination method and tinkering with the hyperparameters
'''

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, RepeatedKFold
from sklearn.feature_selection import RFE, RFECV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt


# Create dataset
def get_dataset():
    x, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
    return x, y

# Create a pipeline
def create_pipeline(mode):
    rfe = RFE(estimator=mode, n_features_to_select=5)
    model = mode
    pipeline = Pipeline(steps=[('s', rfe), ('m', model)])
    return model, pipeline


# Get a list of models to evaluate
def get_models():
    models = dict()
    # Logistic Regression
    rfe = RFE(estimator=LogisticRegression(), n_features_to_select=5)
    model = DecisionTreeClassifier()
    models['lr'] = Pipeline(steps=[('s', rfe), ('m', model)])
    # Perceptron
    rfe = RFE(estimator=Perceptron(), n_features_to_select=5)
    model = DecisionTreeClassifier()
    models['per'] = Pipeline(steps=[('s', rfe), ('m', model)])
    # CART
    model, models['cart'] = create_pipeline(DecisionTreeClassifier())
    # Random Forest
    rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=5)
    model = DecisionTreeClassifier()
    models['rf'] = Pipeline(steps=[('s', rfe), ('m', model)])
    # Gradient Boosting
    rfe = RFE(estimator=GradientBoostingClassifier(), n_features_to_select=5)
    model = DecisionTreeClassifier()
    models['gbm'] = Pipeline(steps=[('s', rfe), ('m', model)])
    return models


# Evaluate a given model using cross-validation
def evaluate_model(model, x, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


# Define classification dataset
xCl, yCl = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# Create classification pipeline
model, pipeline = create_pipeline(DecisionTreeClassifier())
# Evaluate model
n_scores = evaluate_model(model, xCl, yCl)
# Report performance
print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

# Define regression dataset
xReg, yReg = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# Create regression pipeline
model, pipeline = create_pipeline(DecisionTreeRegressor())
# Evaluate model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, xReg, yReg, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
# Report performance
print('\nMAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

# Define dataset
x, y = get_dataset()
# Create pipeline to auto-select features
rfecv = RFECV(estimator=DecisionTreeClassifier())
model = DecisionTreeClassifier()
pipeline = Pipeline(steps=[('s', rfecv), ('m', model)])
# Evaluate model
rfecv_scores = evaluate_model(model, x, y)
# Report performance
print('\nAccuracy: %.3f (%.3f)' % (np.mean(rfecv_scores), np.std(rfecv_scores)))

# Define RFE
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5)
# Fit RFE
rfe.fit(x, y)
# Summarize all features (show which were selected and their rank)
print('\nWhich columns are selected and their ranks:')
for i in range(x.shape[1]):
    print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))

# Get models to evaluate
models = get_models()
# Evaluate models and store results
results, names = list(), list()
print('\nAccuracy of different models using RFE:')
for name, model in models.items():
    scores = evaluate_model(model, x, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
# Plot model performance for comparison
plt.boxplot(results, labels=names, showmeans=True)
plt.show()

