#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 13:27:14 2025

@author: gzen
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv('./src/student_habits_performance.csv')


df.describe()

df.columns

correlations = df.select_dtypes(include='number').corr()

# based on the above, 'exam_score' column has the strongest (positive and negative) correlations
correlations['exam_score'].sort_values()


# netflix_hours           -0.171779   <= Most negative
# social_media_hours      -0.166733
# age                     -0.008907
# attendance_percentage    0.089836
# sleep_hours              0.121683
# exercise_frequency       0.160107
# mental_health_rating     0.321523
# study_hours_per_day      0.825419   <= Most positive
# exam_score               1.000000


#
# STRONG CORRELATIONS
#

# shows pretty strong correlation with scores.
sns.scatterplot(x='study_hours_per_day', y='exam_score', data=df, s=20)
plt.title('Exam Scores vs Study Hours')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.show()


# rating is numeric but discrete categories. Shows a correlation with scores.
sns.barplot(x='mental_health_rating', y='exam_score', data=df)
plt.title('Exam Scores vs Mental Health Rating')
plt.xlabel('Mental Health Rating')
plt.ylabel('Exam Score')
plt.show()


#
# SLIGHT RELATIONSHIP
#

# slight negative correlation between netflix and scores.
sns.scatterplot(x='netflix_hours', y='exam_score', data=df, s=20)
plt.title('Exam Scores vs Netflix Hours')
plt.xlabel('Netflix Hours')
plt.ylabel('Exam Score')
plt.show()


# slight negative correlation betweeen social media and scores.
sns.scatterplot(x='social_media_hours', y='exam_score', data=df, s=20)
plt.title('Exam Scores vs Social Media Hours')
plt.xlabel('Social Media Hours')
plt.ylabel('Exam Score')
plt.show()



# slight correlation between exercise and scores
sns.barplot(x='exercise_frequency', y='exam_score', data=df)
plt.title('Exam Scores vs Exercise Frequency')
plt.xlabel('Exercise Frequency')
plt.ylabel('Exam Score')
plt.show()






#
# LITTLE OR NO EFFECT
#

# Diet quality results in slightly lower scores
sns.boxplot(x='diet_quality', y='exam_score', data=df)
plt.title('Exam Scores vs Diet')
plt.xlabel('Diet Quality')
plt.ylabel('Exam Score')
plt.show()


# Median is slighter higher for bachelors  and slightly lower for masters
sns.boxplot(x='parental_education_level', y='exam_score', data=df)
plt.title('Exam Scores vs Parental Education')
plt.xlabel('Parental Education')
plt.ylabel('Exam Score')
plt.show()



# Internet quality doesn't appear to be a big factor
sns.boxplot(x='internet_quality', y='exam_score', data=df)
plt.title('Exam Scores vs Internet Quality')
plt.xlabel('Internet Quality')
plt.ylabel('Exam Score')
plt.show()



# extracurricular participation doesn't appear to be a big factor
sns.boxplot(x='extracurricular_participation', y='exam_score', data=df)
plt.title('Exam Scores vs Extracurricular')
plt.xlabel('Extracurriculars')
plt.ylabel('Exam Score')
plt.show()


# part time job doesn't appear to be a big factor
sns.boxplot(x='part_time_job', y='exam_score', data=df)
plt.title('Exam Scores vs Part-Time Job')
plt.xlabel('Part-Time Job')
plt.ylabel('Exam Score')
plt.show()


# Gender doesn't seem to have a relationship with scores
sns.barplot(x='gender', y='exam_score', data=df)
plt.title('Exam Scores by Gender')
plt.xlabel('Gender')
plt.ylabel('Exam Score')
plt.show()









