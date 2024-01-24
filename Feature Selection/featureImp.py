'''
Example code using multiple methods (coefficients, decision trees, & permutations) to determine the feature
importance for a made up classification & regression datasets where each only has 5 important variables.
'''

from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor, XGBClassifier
import matplotlib.pyplot as plt


# Method to plot the feature importance for each model below (redundant code)
def plot_feature_importance(importance):
    print() # Add space to make output cleaner
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()


# Method for feature selection
def select_features(x_train, y_train, x_test):
    # Configure to select subset of features
    fs = SelectFromModel(RandomForestClassifier(n_estimators=1000), max_features=5)
    # Learn relationship from training data
    fs.fit(x_train, y_train)
    # Transform train input data
    x_train_fs = fs.transform(x_train)
    # Transform test input data
    x_test_fs = fs.transform(x_test)
    return x_train_fs, x_test_fs, fs



# Define regression dataset
xReg, yReg = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# Define classification dataset
xCl, yCl = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# Define the model
modelR = LinearRegression()
modelC = LogisticRegression()
# Fit the model
modelR.fit(xReg, yReg)
modelC.fit(xCl, yCl)
# Get importance
importance = modelR.coef_
# Summarize importance
plot_feature_importance(importance)
# Run decision tree regressor (CART)
CARTR = DecisionTreeRegressor()
CARTR.fit(xReg, yReg)
# Get importance
importanceCR = CARTR.feature_importances_
# Summarize feature importance
plot_feature_importance(importanceCR)

# Get importance for random forest regressor
RFR = RandomForestRegressor()
RFR.fit(xReg, yReg)
importanceRFR = RFR.feature_importances_
plot_feature_importance(importanceRFR)

# Get importance for XGboost regression features
XGBR = XGBRegressor()
XGBR.fit(xReg, yReg)
importanceXR = XGBR.feature_importances_
plot_feature_importance(importanceXR)

# Get importance for KNN Regression
KNNR = KNeighborsRegressor()
KNNR.fit(xReg, yReg)
# Perform permutation importance
resultsR = permutation_importance(KNNR, xReg, yReg, scoring='neg_mean_squared_error')
importanceKR = resultsR.importances_mean
plot_feature_importance(importanceKR)

# Get importance for Logistic Regression features
importanceLR = modelC.coef_[0]
plot_feature_importance(importanceLR)

# CART feature importance classification
CARTC = DecisionTreeClassifier()
CARTC.fit(xCl, yCl)
# Get importance
importanceCC = CARTC.feature_importances_
# Summarize and plot
plot_feature_importance(importanceCC)

# Same with the random forest classifier
RFC = RandomForestClassifier()
RFC.fit(xCl, yCl)
importanceRFC = RFC.feature_importances_
plot_feature_importance(importanceRFC)

# Get importance for the XGboost classifier
XGBC = XGBClassifier()
XGBC.fit(xCl, yCl)
importanceXC = XGBC.feature_importances_
plot_feature_importance(importanceXC)

# Get importance for KNN classification problem
KNNC = KNeighborsClassifier()
KNNC.fit(xCl, yCl)
resultsC = permutation_importance(KNNC, xCl, yCl, scoring='accuracy')
importanceKC = resultsC.importances_mean
plot_feature_importance(importanceKC)

# Select features based on their importance
# Split classification dataset to train & test splits
x_train, x_test, y_train, y_test = train_test_split(xCl, yCl, test_size=0.33, random_state=1)
# Feature selection
x_train_fs, x_test_fs, fs = select_features(x_train, y_train, x_test)
# Fit model
modelFinal = LogisticRegression(solver='liblinear')
modelFinal.fit(x_train_fs, y_train)
# Evaluate model
yhat = modelFinal.predict(x_test_fs)
# Evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('\nAccuracy: %.2f' % (accuracy * 100))
