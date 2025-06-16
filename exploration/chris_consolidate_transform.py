# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import rcParams
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from scipy.stats import zscore

from scipy.stats import boxcox
from scipy.special import inv_boxcox

import itertools
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, cross_validate

# set random seed for reproducibility
np.random.seed(42)

# allow output to span multiple output lines in the console
pd.set_option('display.max_columns', 500)

# switch to seaborn default stylistic parameters
# see the useful https://seaborn.pydata.org/tutorial/aesthetics.html
sns.set()
sns.set_context('paper') # 'talk' for slightly larger

# change default plot size
rcParams['figure.figsize'] = 9,7

df = pd.read_csv('../src/student_habits_performance.csv', index_col=0)

predictors = ['study_hours_per_day', 'mental_health_rating', 'exercise_frequency', 
              'sleep_hours', 'netflix_hours']  # example predictors
target = 'exam_score'  # example target variable

results = []

def reflective_boxcox_transform(target):
  """
  Apply Box-Cox transformation to the target variable and return the transformed data.
  """
  target_reflected = target.max() - target + 1
  transformed_target, lambda_value = boxcox(target_reflected)
  return transformed_target, lambda_value, (target.max() + 1)

def inverse_reflective_boxcox_transform(transformed_target, lambda_value, max_value):
  """
  Inverse Box-Cox transformation to get back the original target variable.
  """
  reflected_target = inv_boxcox(transformed_target, lambda_value)
  return max_value - reflected_target + 1

def log_transform(predictor):
  """
  Apply log transformation to the predictor variable and return the transformed data.
  """
  return np.log1p(predictor)  # log1p is used to handle zero values safely


X = df[predictors].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Apply Box-Cox transformation to the target variable
y_train_transformed, lambda_value, max_value = reflective_boxcox_transform(y_train)
# Do NOT transform y_test; keep it in original scale for evaluation

# Apply log transformation to the predictors
# X_train_log = np.log1p(X_train[:,4])  # log1p to handle zero values
# X_test_log = np.log1p(X_test[:,4])  # log1p to handle zero values
# X_train = np.delete(X_train, 4, axis=1)  # remove netflix_hours from predictors
# X_test = np.delete(X_test, 4, axis=1)  # remove netflix_hours from predictors
# # Add the log-transformed netflix_hours back to the predictors
# X_train = np.column_stack((X_train, X_train_log))
# X_test = np.column_stack((X_test, X_test_log))

# Apply Polynomial features to the predictors
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Fit a linear regression model
regr = LinearRegression()
regr.fit(X_train_poly, y_train)

# Make predictions on the test set
# y_pred_transformed = regr.predict(X_test)
# y_pred = inverse_reflective_boxcox_transform(y_pred_transformed, lambda_value, max_value)
y_pred = regr.predict(X_test_poly)

# Calculate RMSE and R^2 (both y_test and y_pred are in the original scale)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
results.append({
  'Model': 'Linear Regression (Poly Feat)',
  'RMSE': rmse,
  'R^2': r2,
  'error_ratio': rmse / (df['exam_score'].max() - df['exam_score'].min())
})

# Print results
print("Results:")
for result in results:
    print(f"Model: {result['Model']}, RMSE: {result['RMSE']:.4f}, R^2: {result['R^2']:.4f}, Error Ratio: {result['error_ratio']:.4f}")