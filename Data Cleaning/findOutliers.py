'''
Example code that uses standard deviation & interquartile range methods on a generated Gaussian dataset.
Also uses one-class classification on the Boston Housing dataset.

Extensions include:
- Develop own Gaussian dataset & histogram plot.
- Test IQR method on univariate dataset with non-Gaussian distribution.
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import expon

# Generate Gaussian dataset
# Seed RNG so results are always the same
np.random.seed(1)
# Generate 10,000 univariate observations with a mean of 100 and a std dev of 20
data = 20 * np.random.randn(10000) + 100
# Output histogram of dataset
plt.hist(data)
plt.show()
# Calculate summary statistics
data_mean, data_std = np.mean(data), np.std(data)
print('Finding outliers in a Gaussian dataset using z-scores:')
# Summarize (ensure mean & std dev are correct)
print('\tmean=%.3f stdev=%.3f' % (data_mean, data_std))
# Set cut-off to any value outside 3 std devs
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off
# Identify outliers
outliers = [x for x in data if x < lower or x > upper]
print('\tIdentified outliers: %d' % len(outliers))
# Remove outliers
outliers_rmvd = [x for x in data if x >= lower and x <= upper]
print('\tNon-outlier observations: %d' % len(outliers_rmvd))

# Identify outliers with interquartile range
# Calculate IQR
q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
iqr = q75 - q25
print('\nFind outliers in Gaussian dataset with IQR:')
print('\tPercentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
# Calculate the outlier cutoff
iqr_cut_off = iqr * 1.5
iqr_lower, iqr_upper = q25 - iqr_cut_off, q75 + iqr_cut_off
# Identify outliers
iqr_outliers = [x for x in data if x < iqr_lower or x > iqr_upper]
print('\tIdentified outliers: %d' % len(iqr_outliers))
# Remove outliers
iqr_rmvd = [x for x in data if x >= iqr_lower and x <= iqr_upper]
print('\tNon-outlier observations: %d' % len(iqr_rmvd))

# Identify outliers of non-Gaussian data with IQR
# Create an exponential distribution
exp = expon.rvs(scale=40, size=10000)
# Output as histogram
plt.hist(exp)
plt.show()
# Run IQR on non-Gaussian data
exp25, exp75 = np.percentile(exp, 25), np.percentile(exp, 75)
expIQR = exp75 - exp25
print('\nFind outliers in exponential dataset using IQR:')
print('\tPercentiles: 75th= %.3f, 25th= %.3f, IQR= %.3f' % (exp75, exp25, expIQR))
# Calculate cut-off
exp_cutoff = expIQR * 1.5
exp_lower, exp_upper = exp25 - exp_cutoff, exp75 + exp_cutoff
# Identify outliers
exp_outliers = [x for x in exp if x < exp_lower or x > exp_upper]
print('\tIdentified outliers: %d' % len(exp_outliers))
# Remove outliers
exp_rmvd = [x for x in exp if x >= exp_lower and x <= exp_upper]
print('\tNon-outlier observations: %d' % len(exp_rmvd))

# Identify outliers with one-class classification
# Load Boston Housing dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = pd.read_csv(url, header=None)
# Retrieve the array
df_data = df.values
# Split into input & output elements
x, y = df_data[:, :-1], df_data[:, -1]
# Summarize shape of the dataset
print('\nFind outliers in Boston Housing dataset using one-class classification:')
print('\tOriginal shape: ', x.shape, y.shape)
# Split into train & test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
# Summarize shape of the train & test splits
print('\tTrain-test split shape: ', x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# Identify outliers in the training dataset
lof = LocalOutlierFactor()
yhat = lof.fit_predict(x_train)
# Select all rows that are not outliers
mask = yhat != -1
x_train, y_train = x_train[mask, :], y_train[mask]
# Summarize shape of updated training sets
print('\tNew shape with outliers removed: ',x_train.shape, y_train.shape)
# Fit the model
model = LinearRegression()
model.fit(x_train, y_train)
# Evaluate the model
yhat = model.predict(x_test)
# Evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('\tMAE: %.3f' % mae)



