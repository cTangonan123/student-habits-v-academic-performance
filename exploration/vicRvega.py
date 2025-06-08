#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 18:42:19 2025

@author: victoria
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import zscore


# allow output to span multiple output lines in the console
pd.set_option('display.max_columns', 500)

# switch to seaborn default stylistic parameters
# see the very useful https://seaborn.pydata.org/tutorial/aesthetics.html
sns.set()
# sns.set_context('notebook')   
# sns.set_context('paper')  # smaller
sns.set_context('talk')   # larger

# change default plot size
rcParams['figure.figsize'] = (8, 6)

sns.set_style("whitegrid")

df = pd.read_csv("./src/student_habits_performance.csv")

#Basic overview 
df.shape
df.describe()
df.head()
df.info()

#Missing values
print("Missing values per column:\n", df.isnull().sum())

#numerical columns 
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric columns:", num_cols)

#categorical values
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
print("Categorical columns:", cat_cols)

##What are the means, medians, and quartiles for key numeric features?
print(df[num_cols].describe().T.head())


#Correlation Analysis 
corr = df[num_cols].corr()
corr['exam_score'].sort_values(ascending=False)
plt.figure()
sns.heatmap(corr, cmap="vlag", annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()

###Summary: Which show strongest + and - correlations ###
#Study_hours_per_day, 0.83 (+) correlation, more study = higher scores. 
#metal_health_rating, 0.32, (+) better mental health tends to give higher scores. 
#exercise_frequency	0.12 (+) slightly linked to better scores
#attendance_percentage 0.12	(+) showing up more is mildly helpful
#social_media_hours	–0.17 (-) more social media use ↔ lower scores
#netflix_hours –0.17 (-) more Netflix = lower scores


#Historgram: exam_score
plt.figure()
sns.histplot(df['exam_score'], bins=20, kde=True)
plt.title("Distribution of Exam Score")
plt.xlabel("Exam Score")
plt.show()

#Histogram: study_hours_per_day
plt.figure()
sns.histplot(df['study_hours_per_day'], bins=20, kde=True)
plt.title("Distribution of Study Hours per Day")
plt.xlabel("Study Hours per Day")
plt.show()

#Scatter: study_hours_per_day vs exam_score
plt.figure()
sns.scatterplot(x='study_hours_per_day', y='exam_score', data=df)
plt.title("Study Hours vs. Exam Score")
plt.xlabel("Study Hours per Day")
plt.ylabel("Exam Score")
plt.show()

#####################################################
#POSITIVE correaltions 
#Regression plot: correlation between exam score and study time. 
plt.figure()
sns.regplot(x='study_hours_per_day', y='exam_score', data=df, scatter_kws={'s':20, 'alpha':0.6})
plt.title("Study Time vs. Exam Score (with Fit)")
plt.xlabel("Study Hours per Day")
plt.ylabel("Exam Score")
plt.show()

#Regression plot: Study Hours vs. Exam Score
sns.set_style("whitegrid")
plt.figure()
sns.regplot(x='study_hours_per_day', y='exam_score', data=df,scatter_kws={'s':20, 'alpha':0.6}, line_kws={'color':'darkgreen'})
plt.title("Study Hours per Day vs. Exam Score\nCorrelation: {:.2f}".format(corr.loc['study_hours_per_day','exam_score']))
plt.xlabel("Study Hours per Day")
plt.ylabel("Exam Score")
plt.show()

#Regression plot: Mental Health Rating vs. Exam Score
sns.set_style("whitegrid")
plt.figure()
sns.regplot(x='mental_health_rating', y='exam_score', data=df, scatter_kws={'s':20, 'alpha':0.6}, line_kws={'color':'darkgreen'})
plt.title("Mental Health Rating vs. Exam Score\nCorrelation: {:.2f}".format(corr.loc['mental_health_rating','exam_score']))
plt.xlabel("Mental Health Rating")
plt.ylabel("Exam Score")
plt.show()

#Regression plot: Exercise Frequency vs. Exam Score
sns.set_style("whitegrid")
plt.figure()
sns.regplot(x='exercise_frequency', y='exam_score', data=df, scatter_kws={'s':20, 'alpha':0.6}, line_kws={'color':'darkgreen'})
plt.title("Exercise Frequency vs. Exam Score\nCorrelation: {:.2f}".format(corr.loc['exercise_frequency','exam_score']))
plt.xlabel("Exercise Frequency (per week)")
plt.ylabel("Exam Score")
plt.show()


#####################################################

# NEGATIVE correaltions 
#4. Regression plot: social_media_hours vs exam_score
plt.figure()
sns.regplot(x='social_media_hours', y='exam_score', data=df, scatter_kws={'s':20, 'alpha':0.6},line_kws={'color':'darkblue'})
plt.title("Social Media Hours vs. Exam Score\nCorrelation: {:.2f}".format(corr.loc['social_media_hours','exam_score']))
plt.xlabel("Social Media Hours per Day")
plt.ylabel("Exam Score")
plt.show()

#5. Regression plot: netflix_hours vs exam_score
plt.figure()
sns.regplot(x='netflix_hours', y='exam_score', data=df,scatter_kws={'s':20, 'alpha':0.6}, line_kws={'color':'darkblue'})
plt.title("Netflix Hours vs. Exam Score\nCorrelation: {:.2f}".format(corr.loc['netflix_hours','exam_score']))
plt.xlabel("Netflix Hours per Day")
plt.ylabel("Exam Score")
plt.show()

#####################################################
#Create study‐time groups based on quantiles
breaks = df['study_hours_per_day'].quantile([0, 0.33, 0.66, 1.0])
df['StudyGroup'] = pd.cut(df['study_hours_per_day'], include_lowest=True, bins=breaks, labels=['low', 'medium', 'high'])

#Bar chart of counts by StudyGroup
sns.set_style("whitegrid")
df['StudyGroup'].value_counts().plot.bar()
plt.title("Number of Students by Study Group")
plt.xlabel("Study Group")
plt.ylabel("Count")
plt.show()

#Faceted scatter: sleep_hours vs. exam_score by StudyGroup
sns.set_style("whitegrid")
g = sns.FacetGrid(df, col='StudyGroup', col_order=['low', 'medium', 'high'], height=4, aspect=0.8)
g.map(plt.scatter, 'sleep_hours', 'exam_score', s=20, color='darkred')
plt.show()

#Scatterplot with hue by StudyGroup
sns.set_style("whitegrid")
sns.scatterplot(x='sleep_hours',y='exam_score',data=df, hue='StudyGroup', s=55)
plt.title("Sleep Hours vs. Exam Score by Study Group")
plt.show()


#Histograms of exam_score by StudyGroup
sns.set_style("whitegrid")
g = sns.FacetGrid(df, row='StudyGroup', height=2.5, aspect=1.8)
g.map(plt.hist, 'exam_score', color="darkred")
plt.show()



#####################################################
#Additional possible correaltions 

#violin: exam_score by part_time_job
plt.figure()
sns.violinplot(x='part_time_job', y='exam_score', data=df, inner='box')
plt.title("Exam Score by Part-Time Job")
plt.show()


#barplot: mean exam_score by internet_quality
order = df.groupby('internet_quality')['exam_score'].mean().sort_values().index
plt.figure()
sns.barplot(x='internet_quality', y='exam_score', data=df, order=order)
plt.title("Mean Exam Score by Internet Quality")
plt.show()

#####################################################
#KNN: preliminary work on machine learning; such as test/train dataset split

#features (most positive) as predictors 
features = ['study_hours_per_day', 'mental_health_rating', 'exercise_frequency' ]

X = df[features].values
y = df['exam_score'].values

#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

#scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

#build regressor (using k=5)
regr = KNeighborsRegressor(n_neighbors=5)
regr.fit(X_train, y_train)

#predictions 
predictions = regr.predict(X_test)
print("First 10 predictions: ", predictions[:10])
print("First 10 true values:  ", y_test[:10])

#MSe and RMSE, evaluate 
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mse = mean_squared_error(y_test, predictions)
print(f'MSE (k=5): {mse:.0f}')
print("RMSE (k=5): {:.2f}".format(rmse))

#Blind MSE using y_train
blind_pred = y_train.mean()
mse_blind = mean_squared_error(y_test, [blind_pred]*len(y_test))
print(f'Blind MSE:  {mse_blind:.0f}')
rmse_blind = np.sqrt(mse_blind)
print("Blind RMSE: {:.2f}".format(rmse_blind))

#build regressor (using k=7)
regr7 = KNeighborsRegressor(n_neighbors=7)
regr7.fit(X_train, y_train)
pred7 = regr7.predict(X_test)
mse7 =mean_squared_error(y_test, pred7)
rmse7 = np.sqrt(mean_squared_error(y_test, pred7))
print(f"MSE (k=7): {mse7:.2f}")
print("RMSE (k=7): {:.2f}".format(rmse7))

#build regressor (using k=3)
regr3 = KNeighborsRegressor(n_neighbors=3)
regr3.fit(X_train, y_train)
pred3 = regr3.predict(X_test)
mse3  = mean_squared_error(y_test, pred3)
rmse3 = np.sqrt(mse3)
print(f"MSE (k=3): {mse3:.2f}")
print(f"RMSE (k=3): {rmse3:.2f}")

#Distance weighted KNN (k=5) 
knn_dist = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_dist.fit(X_train, y_train)
pred_dist = knn_dist.predict(X_test)
rmse_dist = np.sqrt(mean_squared_error(y_test, pred_dist))
print("k=5 (distance) → RMSE: {:.2f}".format(rmse_dist))

########################################
# Detect anomalies on the three positive-correlation features

# kNN distance functions
def edist(x, y):
    return np.sqrt(np.sum((x-y)**2))

def dist(x):
    m = x.shape[0]
    dm = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            dm[i, j] = edist(x[i], x[j])
            dm[j, i] = dm[i, j]
    return dm

def knn_anomalies(df, k=7, threshold=2.5):
    # scale the data and compute distance matrix
    x = df.apply(zscore).values
    dm = dist(x)
    
    m = x.shape[0]
    k_distances = np.zeros(m)
    
    for i in range(m):
        row = dm[i, :]
        row_sorted = np.sort(row)
        k_distances[i] = row_sorted[k]
    
    kd_zscores = zscore(k_distances)
    
    anomaly_idxs = np.where(kd_zscores > threshold)[0]
    return anomaly_idxs

#Identify anomalies for + 

anoms = knn_anomalies(df[features], k=7, threshold=2.5)

#Print them out
print(f"Anomalies (k=7, threshold=2.5):")
for idx in anoms:
    st = df.loc[idx, 'study_hours_per_day']
    mh = df.loc[idx, 'mental_health_rating']
    ex = df.loc[idx, 'exercise_frequency']
    score = df.loc[idx, 'exam_score']
    print(f" - index={idx}: study_hours={st}, mental_health={mh}, exercise={ex}, exam_score={score}")

#Add a new 'Anomaly' column to the main df
x = np.full(len(df), False)
x[anoms] = True
df['Anomaly'] = x

df_anom = df[features].copy()
df_anom['Anomaly'] = df['Anomaly']

#Plot all anomalies
sns.pairplot(df_anom, vars=features, hue='Anomaly', diag_kind='kde', palette={False: 'steelblue', True: 'darkred'}, plot_kws={'s':30, 'alpha':0.6})
plt.suptitle("kNN Anomaly Detection on Top Predictors", y=1.02)
plt.show()

#Scatter: Study Hours vs. Exam Score
plt.figure()
sns.scatterplot(x='study_hours_per_day',y='exam_score',data=df,hue='Anomaly',palette={False: 'steelblue', True: 'darkred'},s=40)
plt.title("Exam Score vs. Study Hours")
plt.xlabel("Study Hours per Day")
plt.ylabel("Exam Score")
plt.legend(title='Anomaly', loc='best')
plt.show()

# Mental Health vs. Exam Score
plt.figure()
sns.scatterplot(x='mental_health_rating',y='exam_score',data=df,hue='Anomaly',palette={False: 'steelblue', True: 'darkred'},s=40)
plt.title("Exam Score vs. Mental Health Rating")
plt.xlabel("Mental Health Rating")
plt.ylabel("Exam Score")
plt.legend(title='Anomaly', loc='best')
plt.show()

# Exercise Frequency vs. Exam Score
plt.figure()
sns.scatterplot(x='exercise_frequency',y='exam_score',data=df,hue='Anomaly',palette={False: 'steelblue', True: 'darkred'},s=40)
plt.title("Exam Score vs. Exercise Frequency")
plt.xlabel("Exercise Frequency")
plt.ylabel("Exam Score")
plt.legend(title='Anomaly', loc='best')
plt.show()


####################################################
# Detect anomalies on the two negative-correlation features

features_neg = ['social_media_hours', 'netflix_hours']
df_anom_neg = df[features_neg].copy()
df_scaled_neg = df_anom_neg.apply(zscore)

anoms_neg = knn_anomalies(df_scaled_neg, k=7, threshold=2.5)

#Print anomalous 
print("Anomalies (social_media, netflix; k=7, threshold=2.5):")
for idx in anoms_neg:
    sm = df.loc[idx, 'social_media_hours']
    nf = df.loc[idx, 'netflix_hours']
    sc = df.loc[idx, 'exam_score']
    print(f" - index={idx}: social_media={sm}, netflix={nf}, exam_score={sc}")

# Mark anomalies in the main df
flags = np.full(len(df), False)
flags[anoms_neg] = True
df['Anomaly'] = flags

# Pairplot: two negative features
sns.pairplot(df_anom_neg.assign(Anomaly=flags), vars=features_neg,hue='Anomaly',diag_kind='kde', palette={False: 'steelblue', True: 'darkred'}, plot_kws={'s':30, 'alpha':0.6})
plt.suptitle("Anomaly Detection on Negative Predictors", y=1.02)
plt.show()

#Scatter: Social Media Hours vs. Exam Score
plt.figure()
sns.scatterplot(x='social_media_hours', y='exam_score',data=df,hue='Anomaly', palette={False: 'steelblue', True: 'darkred'},s=50)
plt.title("Exam Score vs. Social Media Hours")
plt.xlabel("Social Media Hours")
plt.ylabel("Exam Score")
plt.show()

#Scatter: Netflix Hours vs. Exam Score
plt.figure()
sns.scatterplot(x='netflix_hours',y='exam_score',data=df,hue='Anomaly',palette={False: 'steelblue', True: 'darkred'},s=50)
plt.title("Exam Score vs. Netflix Hours")
plt.xlabel("Netflix Hours")
plt.ylabel("Exam Score")
plt.show()

####################################################
#KNN Predictions
plt.figure()
plt.scatter(y_test, predictions, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel("True Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.title("KNN Predicted vs True Exam Scores")
plt.show()
