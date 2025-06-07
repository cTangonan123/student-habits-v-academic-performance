# -*- coding: utf-8 -*-
"""
Created on Fri May 23 20:10:43 2025

@author: etorr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from scipy.stats import zscore


# Objective: predict what how a student will do on a exam with given student habits
#Explore what student habit contributes to poor student performance? 
df = pd.read_csv('../Pre-folder/refs/student_habits_performance.csv', low_memory=False)

#Quick overview of data
df.info()


#check unique values
#displays the options for each non-numerical column
#data looks to match its dtype (object, not numerical)

df_no_nums=df.select_dtypes(exclude='number')
df_no_nums.apply(lambda x: len(x.unique()))

#gender
df['gender'].value_counts().head()
#part_time_job 
df['part_time_job'].value_counts().head()
#diet_quality 
df['diet_quality'].value_counts().head()
# parental_education_level 
df['parental_education_level'].value_counts().head()
#internet_quality 
df['internet_quality'].value_counts().head()
#extracurricular_participation
df['extracurricular_participation'].value_counts().head()

#check numerical variables
# what scale is mental health based on? (whole #, decimal, 1/2??)
df.describe(percentiles=[]).round(1)

#check z-score (possible outliers)
# study hours max -3.2 std above, 2.4 std below
# exam score - 3 std below
#social media - 4 std above, 2.1 below
#  netflix - 3.3 above
# attendance 3 below
# Sleep - 2.7 below, 2.9 above
# 
def zscore1(x):
    return(x-x.mean())/x.std()

df2=df.select_dtypes(exclude=['object'])
dfz=df2.apply(zscore1)
x=dfz.describe(percentiles=[]).round(1)

# box plots- confirms outliers
fig, axes = plt.subplots(nrows=3,ncols=3, figsize=(15,10))
df.boxplot(column='age', ax=axes[0,0])
df.boxplot(column='study_hours_per_day', ax=axes[0,1])
df.boxplot(column='social_media_hours', ax=axes[0,2])
df.boxplot(column='netflix_hours', ax=axes[1,0])
df.boxplot(column='attendance_percentage', ax=axes[1,1])
df.boxplot(column='sleep_hours', ax=axes[1,2])
df.boxplot(column='exercise_frequency', ax=axes[2,0])
df.boxplot(column='mental_health_rating',ax=axes[2,1])
df.boxplot(column='exam_score', ax=axes[2,2])
plt.show()

#seaborn boxplots

fig, axes = plt.subplots(nrows=3,ncols=3, figsize=(15,10))
sns.boxplot(y='age',data=df,ax=axes[0,0])
sns.boxplot(y='study_hours_per_day',data=df, ax=axes[0,1])
sns.boxplot(y='social_media_hours',data=df, ax=axes[0,2])
sns.boxplot(y='netflix_hours',data=df, ax=axes[1,0])
sns.boxplot(y='attendance_percentage',data=df, ax=axes[1,1])
sns.boxplot(y='sleep_hours',data=df, ax=axes[1,2])
sns.boxplot(y='exercise_frequency',data=df, ax=axes[2,0])
sns.boxplot(y='mental_health_rating',data=df,ax=axes[2,1])
sns.boxplot(y='exam_score',data=df, ax=axes[2,2])
plt.show()


#histograms -shows more of scale
fig, axes = plt.subplots(nrows=3,ncols=3, figsize=(15,10))
df.hist(column='age', ax=axes[0,0])
df.hist(column='study_hours_per_day', ax=axes[0,1])
df.hist(column='social_media_hours', ax=axes[0,2])
df.hist(column='netflix_hours', ax=axes[1,0])
df.hist(column='attendance_percentage', ax=axes[1,1])
df.hist(column='sleep_hours', ax=axes[1,2])
df.hist(column='exercise_frequency', ax=axes[2,0])
df.hist(column='mental_health_rating',ax=axes[2,1])
df.hist(column='exam_score', ax=axes[2,2])
plt.show()

#histograms -shows more of scale
fig, axes = plt.subplots(nrows=3,ncols=3, figsize=(15,10))
sns.histplot(x='age',data=df, ax=axes[0,0])
sns.histplot(x='study_hours_per_day',data=df, ax=axes[0,1])
sns.histplot(x='social_media_hours',data=df, ax=axes[0,2])
sns.histplot(x='netflix_hours',data=df, ax=axes[1,0])
sns.histplot(x='attendance_percentage',data=df, ax=axes[1,1])
sns.histplot(x='sleep_hours',data=df, ax=axes[1,2])
sns.histplot(x='exercise_frequency',data=df, ax=axes[2,0])
sns.histplot(x='mental_health_rating',data=df,ax=axes[2,1])
sns.histplot(x='exam_score',data=df,ax=axes[2,2])
plt.show()


#number of distinct values -looks okay, mental health rating too small scale? 
ru= df2.apply(lambda x: x.unique().size/x.size)
plt.figure(figsize=(8,4))
ru.plot.bar()
plt.show()

#numeric to categorical
quantiles = df['study_hours_per_day'].quantile([0,0.33,0.66,1])
df['study_level']=pd.cut(df['study_hours_per_day'], 	include_lowest=True, bins=quantiles, labels=['low', 'medium', 'high'])

df['study_level'].value_counts().plot.bar()
plt.show()

quantiles = df['sleep_hours'].quantile([0,0.33,0.66,1])
df['sleep_level']=pd.cut(df['sleep_hours'], 	include_lowest=True, bins=quantiles, labels=['low', 'medium', 'high'])

df['sleep_level'].value_counts().plot.bar()
plt.show()

quantiles = df['attendance_percentage'].quantile([0,0.33,0.66,1])
df['attendance_level']=pd.cut(df['attendance_percentage'], 	include_lowest=True, bins=quantiles, labels=['low', 'medium', 'high'])

df['attendance_level'].value_counts().plot.bar()
plt.show()

#heatmap
# no strong negative corralations
# strong corrleation between exam score and study_hours_per_day
# behavior most correlated with exam score are
# attendance, sleep, exercise, mental health, study hours
corr=df2.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
plt.show()

def most_corr_index(x):
    return x.sort_values(ascending=True).index[1]
df2.corr().apply(most_corr_index)

#density plots
# all normally distributed
fig, axes = plt.subplots(nrows=3,ncols=3, figsize=(15,10))
sns.kdeplot(df['age'], bw=2.0,  ax=axes[0,0])
sns.kdeplot(df['study_hours_per_day'], bw=2.0,  ax=axes[0,1])
sns.kdeplot(df['social_media_hours'], bw=2.0,  ax=axes[0,2])
sns.kdeplot(df['netflix_hours'], bw=2.0,  ax=axes[1,0])
sns.kdeplot(df['attendance_percentage'], bw=2.0,  ax=axes[1,1])
sns.kdeplot(df['sleep_hours'], bw=2.0,  ax=axes[1,2])
sns.kdeplot(df['exercise_frequency'], bw=2.0,  ax=axes[2,0])
sns.kdeplot(df['mental_health_rating'], bw=2.0, ax=axes[2,1])
sns.kdeplot(df['exam_score'], bw=2.0, ax=axes[2,2])
plt.show()

# Question: Does study hours affect exam scores?
sns.scatterplot(x='study_hours_per_day',y='exam_score', data=df)
plt.show()

g=sns.FacetGrid(df, col='study_level', height=4, aspect=0.8)
g.map(plt.hist, 'exam_score', color='blue')
plt.show()

# Question: Does sleep affect exam scores?
sns.scatterplot(x='sleep_hours',y='exam_score', data=df)
plt.show()

# Question: Does exercise affect exam scores?
sns.barplot(x='exercise_frequency',y='exam_score', data=df)
plt.show()

# Question: Does mental health affect exam scores?
sns.barplot(x='mental_health_rating',y='exam_score', data=df)
plt.show()

# Question: Does parent education health affect exam scores?
sns.barplot(x='parental_education_level',y='exam_score', data=df)
plt.show()

# Question: Which gender studies more?
sns.countplot(x='gender',hue='study_level', data=df)
plt.show()

# does the exam score and study hours vary among genders? 
sns.scatterplot(x='exam_score', y='study_hours_per_day', data=df, hue='gender')
plt.show()




