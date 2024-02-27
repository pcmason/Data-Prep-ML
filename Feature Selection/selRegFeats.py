'''
Here use correlation & mutual information to get the top 81 features in a custom regression dataset that has 100 input
features but only 10 of them are relevant. On top of this at the end of the file tune the algorithm to determine what
the best number of features to use are and update the corr & mutual info algorithm.
'''

from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np

# Create method for feature selection
def select_features(x_train, y_train, x_test, method):
    # Configure to select a subset of features
    fs = SelectKBest(score_func=method, k=81)
    # Learn relationship from training data
    fs.fit(x_train, y_train)
    # Transform train input data
    x_train_fs = fs.transform(x_train)
    # Transform test input data
    x_test_fs = fs.transform(x_test)
    return x_train_fs, x_test_fs, fs


# Create the dataset for linear regression problem
x, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
# Split into train & test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)

# Get a baseline performance for the algorithm with no features removed
# Fit the model
model = LinearRegression()
model.fit(x_train, y_train)
# Evaluate the model
yhat = model.predict(x_test)
# Evaluate model efficacy
base_mae = mean_absolute_error(y_test, yhat)
print('Baseline MAE: %.3f' % base_mae)

# Correlation feature selection
x_train_fs, x_test_fs, fs = select_features(x_train, y_train, x_test, f_regression)
# Fit new model on new data
model.fit(x_train_fs, y_train)
# Evaluate correlation model
yhat = model.predict(x_test_fs)
# Plot the correlation scores for the features
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.title('Correlation Scores')
plt.show()
# Evaluate correlation efficacy
corr_mae = mean_absolute_error(y_test, yhat)
print('Correlation MAE: %.3f' % corr_mae)

# Mutual information feature selection
x_train_fs2, x_test_fs2, fs2 = select_features(x_train, y_train, x_test, mutual_info_regression)
# Fit model on mutual info data
mi_model = LinearRegression()
mi_model.fit(x_train_fs2, y_train)
# Evaluate model
yhat = mi_model.predict(x_test_fs2)
# Output mutual information scores for features
plt.bar([i for i in range(len(fs2.scores_))], fs2.scores_)
plt.title('Mutual Information Scores')
plt.show()
# Evaluate mutual info efficacy
mi_mae = mean_absolute_error(y_test, yhat)
print('Mutual Info MAE: %.3f' % mi_mae)

# Optimize the number of features to be trained on for the model
# Define # of features to evaluate [80-100]
num_features = [i for i in range(x.shape[1]-19, x.shape[1]+1)]
# Enumerate each number of features
results = list()
for k in num_features:
    # Create pipeline
    model = LinearRegression()
    fs = SelectKBest(score_func=mutual_info_regression, k=k)
    pipeline = Pipeline(steps=[('sel', fs), ('lr', model)])
    # Evaluate model
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, x, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    results.append(scores)
    # Summarize the results [commented out as results are shown in the graph]
    # print('>%d %.3f (%.3f)' % (k, np.mean(scores), np.std(scores)))

# Plot model performance for visual comparison
plt.boxplot(results, labels=num_features, showmeans=True)
plt.title('Scores Dependent on # of Features Chosen')
plt.show()



