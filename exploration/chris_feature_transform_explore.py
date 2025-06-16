"""
In this script, I'm exploring based on skewness of the features and target variable, exam_score.
- applying various transformations to the features and target variable to improve the linear regression model's performance.
- using Box-Cox transformation for the target variable, and log and square root transformations for the predictors. Comparing between
  the original and transformed features to see if the transformations improve the fit of the linear 
  regression model.
"""


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
from sklearn.metrics import mean_squared_error
from scipy.stats import zscore


# allow output to span multiple output lines in the console
pd.set_option('display.max_columns', 500)

# switch to seaborn default stylistic parameters
# see the useful https://seaborn.pydata.org/tutorial/aesthetics.html
sns.set()
sns.set_context('paper') # 'talk' for slightly larger

# change default plot size
rcParams['figure.figsize'] = 9,7

df = pd.read_csv('../src/student_habits_performance.csv', index_col=0)

sns.pairplot(df.select_dtypes(include=[np.number]), kind='reg', diag_kind='kde', plot_kws={'line_kws':{'color':'red'}})
plt.show()

"""
Box-cox is another type of transformation for fitting 
"""
# Box-Cox Transformation for exam_score testing to see distribution
from scipy.stats import boxcox
from scipy.special import inv_boxcox

# Add a small constant if there are zeros
exam_score_reflected = df['exam_score'].max() + 1 - df['exam_score']
exam_score_bc, fitted_lambda = boxcox(exam_score_reflected + 1e-6)
df['exam_score_reflected_boxcox'] = exam_score_bc

print(f"Box-Cox lambda for exam_score: {fitted_lambda:.3f}")

sns.histplot(df['exam_score_reflected_boxcox'], bins=20, kde=True)
plt.title("Box-Cox Transformed Exam Score")
plt.xlabel("Transformed Exam Score")
plt.show()

# Linear Regression with exam_score
col_pred = df.select_dtypes(include='number').columns
col_pred = col_pred.drop(['exam_score', 'exam_score_reflected_boxcox'])
lin_models = pd.DataFrame(columns=['feature', 'R^2', 'RMSE', 'intercept', 'coefficients'])

n_cols = 2
n_rows = math.ceil(len(df.select_dtypes(include='number').columns) / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten()

for i, c in enumerate(col_pred):
  X = df[[c]].values.reshape(-1,1)
  y = df['exam_score'].values

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  reg = LinearRegression()
  reg.fit(X_train, y_train)

  prediction = reg.predict(X_test)
  r2 = reg.score(X_test, y_test)
  RMSE = np.sqrt(np.mean((y_test - prediction)**2))

  x_vals = np.linspace(min(X.flatten()), max(X.flatten()), 100)
  y_vals = reg.intercept_ + reg.coef_[0]*x_vals

  sns.scatterplot(x=X.flatten(), y=y, ax=axes[i])
  axes[i].plot(x_vals, y_vals, color='darkred')
  axes[i].set_title(f'Linear Regression: exam_score by {c}\nR^2: {r2:.3f}, RMSE: {RMSE:.3f}, Coeff: {reg.coef_[0]:.3f}, Intercept: {reg.intercept_:.3f}')
  axes[i].set_xlabel(c)
  axes[i].set_ylabel('exam_score')
  axes[i].grid()
  new_row = pd.DataFrame([[c, r2, RMSE, reg.intercept_, reg.coef_[0]]], columns=['feature', 'R^2', 'RMSE', 'intercept', 'coefficients'])
  lin_models = pd.concat([lin_models, new_row], ignore_index=True)
# Hide any unused subplots
for j in range(i + 1, len(axes)):
  fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

sorted_lin_models = lin_models.sort_values(by='R^2', ascending=False)
print("Linear Regression Models Summary:")
print(sorted_lin_models)

# test with reverse Box-Cox transformed exam_score
lin_models_bc = pd.DataFrame(columns=['feature', 'R^2', 'RMSE', 'intercept', 'coefficients'])

n_cols = 2
n_rows = math.ceil(len(df.select_dtypes(include='number').columns) / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten()

for i, c in enumerate(col_pred):
  X = df[[c]].values.reshape(-1,1)
  y = df['exam_score_reflected_boxcox'].values

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  reg = LinearRegression()
  reg.fit(X_train, y_train)

  prediction = reg.predict(X_test)
  r2 = reg.score(X_test, y_test)
  RMSE = np.sqrt(np.mean((y_test - prediction)**2))

  """
  exam_score_reflected = df['exam_score'].max() + 1 - df['exam_score']
  exam_score_reflected - df['exam_score'].max() - 1 = -df['exam_score']
  df['exam_score'] = df['exam_score'].max() + 1 - exam_score_reflected
  Reflecting the Box-Cox transformed values back to the original scale
  y_rev = y.max() + 1 - y
  y = - y_rev + y.max() + 1
  """
  x_vals = np.linspace(min(X.flatten()), max(X.flatten()), 100)
  y_vals = (inv_boxcox(reg.intercept_ + reg.coef_[0]*x_vals, fitted_lambda) - 1e-6)
  y_vals = df['exam_score'].max() + 1 - y_vals  # Reflect back to original scale
  sns.scatterplot(x=X.flatten(), y=df['exam_score'], ax=axes[i])
  axes[i].plot(x_vals, y_vals, color='darkred')
  axes[i].set_title(f'Box-Cox Linear Regression: exam_score by {c}\nR^2: {r2:.3f}, RMSE: {RMSE:.3f}, Coeff: {reg.coef_[0]:.3f}, Intercept: {reg.intercept_:.3f}')
  axes[i].set_xlabel(c)
  axes[i].set_ylabel('exam_score (Box-Cox Transformed)')
  axes[i].grid()

  # Append results to the DataFrame
  new_row = pd.DataFrame([[c, r2, RMSE, reg.intercept_, reg.coef_[0]]], columns=['feature', 'R^2', 'RMSE', 'intercept', 'coefficients'])
  lin_models_bc = pd.concat([lin_models_bc, new_row], ignore_index=True)

# Hide any unused subplots
for j in range(i + 1, len(axes)):
  fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

sorted_lin_models_bc = lin_models_bc.sort_values(by='R^2', ascending=False)
print("Linear Regression Models with Box-Cox Transformed Exam Score Summary:")
print(sorted_lin_models_bc)

"""
So there is a small improvement of fit with the reflected-Box-Cox transformed on the target exam_score,

"""

# Individual predictor transformations
# left_skewed features include: 'attendance_percentage' **(ATTEMPTED, not worth it.)**
# right_skewed features include: 'netflix_hours'**(ATTEMPTED, not worth it), 'social_media_hours', maybe 'study_hours_per_day'
# 'study_hours_per_day' is not skewed, but has a non-linear relationship with exam_score

# Transforming 'attendance_percentage' using Box-Cox
attendance_reflected = df['attendance_percentage'].max() + 1 - df['attendance_percentage']
attendance_bc, attendance_lambda = boxcox(df['attendance_percentage'])
attendance_bc, fitted_lambda = boxcox(attendance_reflected + 1e-6)
df['attendance_reflected_boxcox'] = attendance_bc

print(f"Box-Cox lambda for attendance_reflected: {attendance_lambda:.3f}")

sns.histplot(df['attendance_reflected_boxcox'], bins=20, kde=True)
plt.title("Box-Cox Transformed Attendance Percentage")
plt.xlabel("Transformed Attendance Percentage")
plt.show()
sns.histplot(df['attendance_percentage'], bins=20, kde=True)
plt.title("Original Attendance Percentage")
plt.xlabel("Original Attendance Percentage")
plt.show()

# Linear Regression with Box-Cox transformed attendance_percentage
X = df[['attendance_reflected_boxcox']].values.reshape(-1, 1)
y = df['exam_score_reflected_boxcox'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)

prediction = reg.predict(X_test)
r2 = reg.score(X_test, y_test)
RMSE = np.sqrt(np.mean((y_test - prediction)**2))

x_vals = np.linspace(min(df['attendance_percentage']), max(df['attendance_percentage']), 100)
y_vals = (inv_boxcox(reg.intercept_ + reg.coef_[0]*x_vals, attendance_lambda)) - 1e-6
y_vals = df['attendance_percentage'].max() + 1 - y_vals  # Reflect back to original scale

sns.scatterplot(x=df['attendance_percentage'], y=df['exam_score'], label='Actual', color='blue')
plt.plot(x_vals, y_vals, color='darkred')
plt.title(f'Linear Regression: Exam Score by Attendance Percentage (Box-Cox Transformed)\nR^2: {r2:.3f}, RMSE: {RMSE:.3f}, Coeff: {reg.coef_[0]:.3f}, Intercept: {reg.intercept_:.3f}')
plt.xlabel('Attendance Percentage (Box-Cox Transformed)')
plt.ylabel('Exam Score')
plt.grid()
plt.legend()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(df[['attendance_percentage']], df['exam_score'], test_size=0.3, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
prediction = reg.predict(X_test)
r2 = reg.score(X_test, y_test)
RMSE = np.sqrt(np.mean((y_test - prediction)**2))
x_vals = np.linspace(min(df['attendance_percentage']), max(df['attendance_percentage']), 100)
y_vals = reg.intercept_ + reg.coef_[0]*x_vals

sns.scatterplot(x=df['attendance_percentage'], y=df['exam_score'], label='Actual', color='blue')
plt.plot(x_vals, y_vals, color='darkred')
plt.title(f'Linear Regression: Exam Score by Attendance Percentage\nR^2: {r2:.3f}, RMSE: {RMSE:.3f}, Coeff: {reg.coef_[0]:.3f}, Intercept: {reg.intercept_:.3f}')
plt.xlabel('Attendance Percentage')
plt.ylabel('Exam Score')
plt.grid()
plt.legend()
plt.show()

"""
RIGHT SKEWED FEATURES
"""
# Transforming 'netflix_hours' using log transformation ->  Winner
df['netflix_hours_log'] = np.log1p(df['netflix_hours'] + 1)
sns.histplot(df['netflix_hours_log'], bins=20, kde=True)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(df['netflix_hours'], bins=20, kde=True, ax=axes[0])
sns.histplot(df['netflix_hours_log'], bins=20, kde=True, ax=axes[1])
axes[0].set_title("Original Netflix Hours")
axes[1].set_title("Log Transformed Netflix Hours")
plt.show()

# Transforming 'nextflix_hours' using square root transformation
df['netflix_hours_sqrt'] = np.sqrt(df['netflix_hours'])

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(df['netflix_hours'], bins=20, kde=True, ax=axes[0])
sns.histplot(df['netflix_hours_sqrt'], bins=20, kde=True, ax=axes[1])
axes[0].set_title("Original Netflix Hours")
axes[1].set_title("Square Root Transformed Netflix Hours")
plt.show()

# Linear Regression with log transformed netflix_hours
X = df[['netflix_hours_log']].values.reshape(-1, 1)
y = df['exam_score'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
prediction = reg.predict(X_test)
r2 = reg.score(X_test, y_test)
RMSE = np.sqrt(np.mean((y_test - prediction)**2))

fig, axes = plt.subplots(1,2,figsize=(10, 6))
axes.flatten()

x_vals = np.linspace(min(df['netflix_hours']), max(df['netflix_hours']), 100)
y_vals = reg.intercept_ + reg.coef_[0]*np.log1p(x_vals + 1)

sns.scatterplot(x=prediction, y=y_test, label='Actual', color='blue', ax=axes[0])
axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='darkred', linestyle='--')
axes[0].set_title(f'Linear Regression: Exam Score by Netflix Hours (Log Transformed predictor)\nPredicted vs Actual\nR^2: {r2:.3f}, RMSE: {RMSE:.3f}, Coeff: {reg.coef_[0]:.3f}, Intercept: {reg.intercept_:.3f}')
axes[0].set_xlabel('Netflix Hours (Log Transformed)')
axes[0].set_ylabel('Exam Score')
axes[0].grid()
axes[0].legend()


sns.scatterplot(x=df['netflix_hours'], y=df['exam_score'], label='Actual', color='blue', ax=axes[1])
axes[1].plot(x_vals, y_vals, color='darkred')
axes[1].set_title(f'Linear Regression: Exam Score by Netflix Hours (Log Transformed)\nR^2: {r2:.3f}, RMSE: {RMSE:.3f}, Coeff: {reg.coef_[0]:.3f}, Intercept: {reg.intercept_:.3f}')
axes[1].set_xlabel('Netflix Hours (Log Transformed)')
axes[1].set_ylabel('Exam Score')
axes[1].grid()
axes[1].legend()
plt.show()

# Linear Regression with log transformed netflix_hours and Box-Cox transformed exam_score
X = df[['netflix_hours_log']].values.reshape(-1, 1)
y = df['exam_score_reflected_boxcox'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
prediction = reg.predict(X_test)
r2 = reg.score(X_test, y_test)
RMSE = np.sqrt(np.mean((y_test - prediction)**2))

fig, axes = plt.subplots(1, 2, figsize=(10, 6))
axes = axes.flatten()

sns.scatterplot(x=prediction, y=y_test, label='Actual', color='blue', ax=axes[0])
axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='darkred', linestyle='--')
axes[0].set_title(f'Linear Regression: Exam Score by Netflix Hours (Log Transformed predictor, Box-Cox transformed target)\nPredicted vs Actual\nR^2: {r2:.3f}, RMSE: {RMSE:.3f}, Coeff: {reg.coef_[0]:.3f}, Intercept: {reg.intercept_:.3f}')
axes[0].set_xlabel('Netflix Hours (Log Transformed)')
axes[0].set_ylabel('Exam Score (Box-Cox Transformed)')
axes[0].grid()

x_vals = np.linspace(min(df['netflix_hours']), max(df['netflix_hours']), 100)
y_vals = (inv_boxcox(reg.intercept_ + reg.coef_[0]*np.log1p(x_vals + 1), fitted_lambda)) - 1e-6
y_vals = df['exam_score'].max() + 1 - y_vals  # Reflect back to original scale

sns.scatterplot(x=df['netflix_hours'], y=df['exam_score'], label='Actual', color='blue', ax=axes[1])
axes[1].plot(x_vals, y_vals, color='darkred')
axes[1].set_title(f'Linear Regression: Exam Score by Netflix Hours (Log Transformed, Box-Cox)\nR^2: {r2:.3f}, RMSE: {RMSE:.3f}, Coeff: {reg.coef_[0]:.3f}, Intercept: {reg.intercept_:.3f}')
axes[1].set_xlabel('Netflix Hours (Log Transformed)')
axes[1].set_ylabel('Exam Score (Box-Cox Transformed)')
axes[1].grid()
axes[1].legend()
plt.show()

# Linear Regression with square root transformed netflix_hours
X = df[['netflix_hours_sqrt']].values.reshape(-1, 1)
y = df['exam_score'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
prediction = reg.predict(X_test)
r2 = reg.score(X_test, y_test)
RMSE = np.sqrt(np.mean((y_test - prediction)**2))
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
axes = axes.flatten()

x_vals = np.linspace(min(df['netflix_hours']), max(df['netflix_hours']), 100)
y_vals = reg.intercept_ + reg.coef_[0]*np.sqrt(x_vals)

sns.scatterplot(x=prediction, y=y_test, label='Actual', color='blue', ax=axes[0])
axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='darkred', linestyle='--')
axes[0].set_title(f'Linear Regression: Exam Score by Netflix Hours (Square Root Transformed predictor)\nPredicted vs Actual\nR^2: {r2:.3f}, RMSE: {RMSE:.3f}, Coeff: {reg.coef_[0]:.3f}, Intercept: {reg.intercept_:.3f}')
axes[0].set_xlabel('Netflix Hours (Square Root Transformed)')
axes[0].set_ylabel('Exam Score')
axes[0].grid()

x_vals = np.linspace(min(df['netflix_hours']), max(df['netflix_hours']), 100)
y_vals = reg.intercept_ + reg.coef_[0]*np.sqrt(x_vals)

sns.scatterplot(x=df['netflix_hours'], y=df['exam_score'], label='Actual', color='blue', ax=axes[1])
axes[1].plot(x_vals, y_vals, color='darkred')
axes[1].set_title(f'Linear Regression: Exam Score by Netflix Hours (Square Root Transformed)\nR^2: {r2:.3f}, RMSE: {RMSE:.3f}, Coeff: {reg.coef_[0]:.3f}, Intercept: {reg.intercept_:.3f}')
axes[1].set_xlabel('Netflix Hours (Square Root Transformed)')
axes[1].set_ylabel('Exam Score')
axes[1].grid()
axes[1].legend()
plt.show()

# Linear Regression with square root transformed netflix_hours and Box-Cox transformed exam_score
X = df[['netflix_hours_sqrt']].values.reshape(-1, 1)
y = df['exam_score_reflected_boxcox'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
prediction = reg.predict(X_test)
r2 = reg.score(X_test, y_test)
RMSE = np.sqrt(np.mean((y_test - prediction)**2))
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
axes = axes.flatten()

x_vals = np.linspace(min(df['netflix_hours']), max(df['netflix_hours']), 100)
y_vals = (inv_boxcox(reg.intercept_ + reg.coef_[0]*np.sqrt(x_vals), fitted_lambda)) - 1e-6
y_vals = df['exam_score'].max() + 1 - y_vals  # Reflect back to original scale

sns.scatterplot(x=prediction, y=y_test, label='Actual', color='blue', ax=axes[0])
axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='darkred', linestyle='--')
axes[0].set_title(f'Linear Regression: Exam Score by Netflix Hours (Square Root Transformed predictor, Box-Cox transformed target)\nPredicted vs Actual\nR^2: {r2:.3f}, RMSE: {RMSE:.3f}, Coeff: {reg.coef_[0]:.3f}, Intercept: {reg.intercept_:.3f}')
axes[0].set_xlabel('Netflix Hours (Square Root Transformed)')
axes[0].set_ylabel('Exam Score (Box-Cox Transformed)')
axes[0].grid()

sns.scatterplot(x=df['netflix_hours'], y=df['exam_score'], label='Actual', color='blue', ax=axes[1])
axes[1].plot(x_vals, y_vals, color='darkred')
axes[1].set_title(f'Linear Regression: Exam Score by Netflix Hours (Square Root Transformed, Box-Cox)\nR^2: {r2:.3f}, RMSE: {RMSE:.3f}, Coeff: {reg.coef_[0]:.3f}, Intercept: {reg.intercept_:.3f}')
axes[1].set_xlabel('Netflix Hours (Square Root Transformed)')
axes[1].set_ylabel('Exam Score (Box-Cox Transformed)')
axes[1].grid()
axes[1].legend()
plt.show()

"""
So the log transformation of netflix_hours is the better predictor of exam_score, with a small improvement in fit with the Box-Cox transformed exam_score.
The square root transformation of netflix_hours is not as good as the log transformation, but still provides a reasonable fit with a lower RMSE than the original netflix_hours.
"""

# Transforming 'Study Hours per Day' using log transformation
df['study_hours_per_day_log'] = np.log1p(df['study_hours_per_day'] + 1)
sns.histplot(df['study_hours_per_day_log'], bins=20, kde=True)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(df['study_hours_per_day'], bins=20, kde=True, ax=axes[0])
sns.histplot(df['study_hours_per_day_log'], bins=20, kde=True, ax=axes[1])
axes[0].set_title("Original Study Hours per Day")
axes[1].set_title("Log Transformed Study Hours per Day")
plt.show()

# Linear Regression with log transformed study_hours_per_day
X = df[['study_hours_per_day_log']].values.reshape(-1, 1)
y = df['exam_score'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
prediction = reg.predict(X_test)
r2 = reg.score(X_test, y_test)
RMSE = np.sqrt(np.mean((y_test - prediction)**2))
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
axes = axes.flatten()
x_vals = np.linspace(min(df['study_hours_per_day']), max(df['study_hours_per_day']), 100)
y_vals = reg.intercept_ + reg.coef_[0]*np.log1p(x_vals + 1)

sns.scatterplot(x=prediction, y=y_test, label='Actual', color='blue', ax=axes[0])
axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='darkred', linestyle='--')
axes[0].set_title(f'Linear Regression: Exam Score by Study Hours per Day (Log Transformed predictor)\nPredicted vs Actual\nR^2: {r2:.3f}, RMSE: {RMSE:.3f}, Coeff: {reg.coef_[0]:.3f}, Intercept: {reg.intercept_:.3f}')
axes[0].set_xlabel('Study Hours per Day (Log Transformed)')
axes[0].set_ylabel('Exam Score')
axes[0].grid()

sns.scatterplot(x=df['study_hours_per_day'], y=df['exam_score'], label='Actual', color='blue', ax=axes[1])
axes[1].plot(x_vals, y_vals, color='darkred')
axes[1].set_title(f'Linear Regression: Exam Score by Study Hours per Day (Log Transformed)\nR^2: {r2:.3f}, RMSE: {RMSE:.3f}, Coeff: {reg.coef_[0]:.3f}, Intercept: {reg.intercept_:.3f}')
axes[1].set_xlabel('Study Hours per Day (Log Transformed)')
axes[1].set_ylabel('Exam Score')
axes[1].grid()
axes[1].legend()
plt.show()

# Linear Regression with log transformed study_hours_per_day and Box-Cox transformed exam_score
X = df[['study_hours_per_day_log']].values.reshape(-1, 1)
y = df['exam_score_reflected_boxcox'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
prediction = reg.predict(X_test)
r2 = reg.score(X_test, y_test)
RMSE = np.sqrt(np.mean((y_test - prediction)**2))

fig, axes = plt.subplots(1, 2, figsize=(10, 6))
axes = axes.flatten()

x_vals = np.linspace(min(df['study_hours_per_day']), max(df['study_hours_per_day']), 100)
y_vals = (inv_boxcox(reg.intercept_ + reg.coef_[0]*np.log1p(x_vals + 1), fitted_lambda)) - 1e-6
y_vals = df['exam_score'].max() + 1 - y_vals  # Reflect back to original scale

sns.scatterplot(x=prediction, y=y_test, label='Actual', color='blue', ax=axes[0])
axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='darkred', linestyle='--')
axes[0].set_title(f'Linear Regression: Exam Score by Study Hours per Day (Log Transformed predictor, Box-Cox transformed target)\nPredicted vs Actual\nR^2: {r2:.3f}, RMSE: {RMSE:.3f}, Coeff: {reg.coef_[0]:.3f}, Intercept: {reg.intercept_:.3f}')
axes[0].set_xlabel('Study Hours per Day (Log Transformed)')
axes[0].set_ylabel('Exam Score (Box-Cox Transformed)')

sns.scatterplot(x=df['study_hours_per_day'], y=df['exam_score'], label='Actual', color='blue', ax=axes[1])
axes[1].plot(x_vals, y_vals, color='darkred')
axes[1].set_title(f'Linear Regression: Exam Score by Study Hours per Day (Log Transformed, Box-Cox)\nR^2: {r2:.3f}, RMSE: {RMSE:.3f}, Coeff: {reg.coef_[0]:.3f}, Intercept: {reg.intercept_:.3f}')
axes[1].set_xlabel('Study Hours per Day (Log Transformed)')
axes[1].set_ylabel('Exam Score (Box-Cox Transformed)')
axes[1].grid()
axes[1].legend()
plt.show()

# Linear Regression with square root transformed study_hours_per_day
df['study_hours_per_day_sqrt'] = np.sqrt(df['study_hours_per_day'])
X = df[['study_hours_per_day_sqrt']].values.reshape(-1, 1)
y = df['exam_score'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)

prediction = reg.predict(X_test)
r2 = reg.score(X_test, y_test)
RMSE = np.sqrt(np.mean((y_test - prediction)**2))

fig, axes = plt.subplots(1, 2, figsize=(10, 6))
axes = axes.flatten()

x_vals = np.linspace(min(df['study_hours_per_day']), max(df['study_hours_per_day']), 100)
y_vals = reg.intercept_ + reg.coef_[0]*np.sqrt(x_vals)

sns.scatterplot(x=prediction, y=y_test, label='Actual', color='blue', ax=axes[0])
axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='darkred', linestyle='--')
axes[0].set_title(f'Linear Regression: Exam Score by Study Hours per Day (Square Root Transformed predictor)\nPredicted vs Actual\nR^2: {r2:.3f}, RMSE: {RMSE:.3f}, Coeff: {reg.coef_[0]:.3f}, Intercept: {reg.intercept_:.3f}')
axes[0].set_xlabel('Study Hours per Day (Square Root Transformed)')
axes[0].set_ylabel('Exam Score')
axes[0].grid()

sns.scatterplot(x=df['study_hours_per_day'], y=df['exam_score'], label='Actual', color='blue', ax=axes[1])
axes[1].plot(x_vals, y_vals, color='darkred')
axes[1].set_title(f'Linear Regression: Exam Score by Study Hours per Day (Square Root Transformed)\nR^2: {r2:.3f}, RMSE: {RMSE:.3f}, Coeff: {reg.coef_[0]:.3f}, Intercept: {reg.intercept_:.3f}')
axes[1].set_xlabel('Study Hours per Day (Square Root Transformed)')
axes[1].set_ylabel('Exam Score')
axes[1].grid()
axes[1].legend()
plt.show()

"""
Linear Regression with either log or square root transformed study_hours_per_day does not improve the fit compared to the original study_hours_per_day.
"""

# Attempting to consolidate information
"""
- Tranformations applied:
  - Box-Cox on exam_score and attendance_percentage
  - Log transformation on netflix_hours and study_hours_per_day
  - Square root transformation on netflix_hours and study_hours_per_day
- Best predictor of exam_score:
  - Log transformed netflix_hours with Box-Cox transformed exam_score
- Other predictors:
  - Attendance percentage (Box-Cox transformed)
  - Study hours per day (log transformed)
  - Netflix hours (log transformed)
- Square root transformation of netflix_hours and study_hours_per_day did not improve the fit.
- Linear regression models were built for each transformation and predictor.
- The log transformation of netflix_hours provided the best fit for predicting exam_score, with a small improvement when using Box-Cox transformed exam_score.
"""

import itertools
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import KFold


predictors = ['study_hours_per_day', 'mental_health_rating', 'exercise_frequency', 
              'sleep_hours', 'netflix_hours_log', 'social_media_hours']  # example predictors
target = 'exam_score_reflected_boxcox'  # example target variable

results = []

for k in range(1, len(predictors)+1):
    for combo in itertools.combinations(predictors, k):
        #
        X = df[list(combo)].values
        y = df[target].values
        sub_results = []
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Apply cross validation
        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        for train_idx, test_idx in cv.split(X_train):
            X_cv_train, X_cv_test = X_train[train_idx], X_train[test_idx]
            y_cv_train, y_cv_test = y_train[train_idx], y_train[test_idx]
            model = LinearRegression()
            model.fit(X_cv_train, y_cv_train)
            y_cv_pred = model.predict(X_cv_test)
            cv_rmse = root_mean_squared_error(y_cv_test, y_cv_pred)
            cv_r2 = r2_score(y_cv_test, y_cv_pred)
            sub_results.append({
                'predictors': combo,
                'rmse': cv_rmse,
                'r2': cv_r2
            })
        results.append({
            'predictors': combo,
            'rmse-mean': np.mean([res['rmse'] for res in sub_results]),
            'r2-mean': np.mean([res['r2'] for res in sub_results])
        })
        # model = LinearRegression()
        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)
        # rmse = mean_squared_error(y_test, y_pred, squared=False)
        # r2 = r2_score(y_test, y_pred)
        # results.append({
        #     'predictors': combo,
        #     'rmse': rmse,
        #     'r2': r2
        # })


results_sorted = sorted(results, key=lambda x: x['r2-mean'], reverse=True)
for res in results_sorted[:10]:  # Show top 10
    print(f"Predictors: {res['predictors']}, r2 avg: {res['r2-mean']:.3f}, RMSE avg: {res['rmse-mean']:.2f}")


top_combo = results_sorted[0]['predictors']
print(f"Best combination of predictors: {top_combo}")

# Train a linear regression model with the best predictors
X_best = df[list(top_combo)]
y_best = df[target]
X_train, X_test, y_train, y_test = train_test_split(X_best, y_best, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# Evaluate the model
rmse_best = root_mean_squared_error(y_test, y_pred)
r2_best = r2_score(y_test, y_pred)
print(f"Best Model RMSE: {rmse_best:.2f}, R²: {r2_best:.3f}")

# Plotting the best model predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("True Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.title("Best Model: Predicted vs True Exam Scores (Top Predictors)\nRMSE: {:.2f}, R^2: {:.3f}".format(rmse_best, r2_best))
plt.grid()
plt.tight_layout()
plt.show()


# Implementation of PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
# Create polynomial features for the best predictors
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_best)
# Split the data
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y_best, test_size=0.3, random_state=42)
# Scale the polynomial features
scaler_poly = StandardScaler()
X_train_poly = scaler_poly.fit_transform(X_train_poly)
X_test_poly = scaler_poly.transform(X_test_poly)

# Train a linear regression model with polynomial features
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train_poly)
y_poly_pred = poly_reg.predict(X_test_poly)
# Evaluate the polynomial model
rmse_poly = root_mean_squared_error(y_test_poly, y_poly_pred)
r2_poly = r2_score(y_test_poly, y_poly_pred)
print(f"Polynomial Model RMSE: {rmse_poly:.2f}, R²: {r2_poly:.3f}")
# Plotting the polynomial model predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test_poly, y_poly_pred, alpha=0.6)
plt.plot([y_test_poly.min(), y_test_poly.max()], [y_test_poly.min(), y_test_poly.max()], 'k--', lw=2)
plt.xlabel("True Exam Scores")
plt.ylabel("Predicted Exam Scores (Polynomial Features)")
plt.title("Polynomial Model: Predicted vs True Exam Scores\nRMSE: {:.2f}, R^2: {:.3f}".format(rmse_poly, r2_poly))
plt.grid()
plt.tight_layout()
plt.show()
# Summary of the best model
print("Best Model Summary:")
print(f"Predictors: {top_combo}")
print(f"RMSE: {rmse_best:.2f}, R²: {r2_best:.3f}")
print(f'{rmse_best/(df["exam_score"].max() - df["exam_score"].min()):.2%} of the range of exam_score')