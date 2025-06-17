# Student Habits vs Academic Performance

## Introduction 

Students usually have busy lives while enrolled in school. Many of their habits can impact their performance in class, especially when taking an exam. In this project, we aim to
determine a relationship between a student’s habits and potentially predict their exam score based on those habits. Habits, such as a student's sleep hours, study time, attendance
percentage, exercise frequency, and extracurricular participation, will be analyzed. We will be using the “Student Habits vs Academic Performance” dataset from Kaggle. First, we’ll load
and clean the data using pandas (checking for nulls and performing basic sanity checks), then explore relationships using matplotlib (histograms, scatterplots, and correlation). As we
complete additional class modules on data wrangling, kNN, and linear regression, we expect to utilize further methods to analyze our data more effectively and predict student
performance. We will interpret the results to answer our question about the relationship between the students' habits and their final score performance.


## Selection Of Data

We will be using the “Student Habits vs Academic Performance” dataset, which was obtained from Kaggle. It was downloaded on May 25th at 7:30 p.m., and the last update was approximately
a month prior to our download. The author of the data is Jayantha Nath. The data is simulated, consisting of 1000 synthetic student records. There are a total of 1000 rows and 16 
columns. Each student is assigned a unique student ID, along with their age and gender, to identify them in columns 0, 1, and 2. Columns 3 through 14 represent the habits that students
can partake in that can affect their exam scores. These habits range from the hours spent studying to the number of hours spent watching Netflix.  Exam scores are given in the last
column. After preprocessing the data, we noticed that there were 91 missing values for the categorical column, parental_education_level. These values were replaced with the value
‘unknown’. We also noticed that outliers existed in the columns' study_hours_per_day' and 'social_media_hours'. 

## Methods

For this project, we used Python as our programming language, along with the following libraries:

-pandas and numpy for data manipulation
-seaborn and matplotlib for data visualization
-math for basic mathematical operations
-scikit-learn (sklearn) for data modeling, training, and evaluation
-scipy for scientific computing

Jupyter Notebook was used by all group members in their local Integrated Development Environments (IDEs). Each group member contributed their own visualizations and analysis of the
data, along with their thoughts on the structure and interpretation of the dataset. Responsibilities were divided across the group—for example, one member handled data cleaning, another
worked on KNN regression, and others focused on modeling and evaluation. Tasks were assigned and reviewed during our weekly meetings.

After examining the pair plots in the "Understanding the Numerical Data" section and the heatmap in the Correlation Analysis section, we observed several strong linear relationships
between student habits and exam scores. This motivated the use of Linear Regression, as it is well-suited for modeling linear trends. However, for some features, the relationships were
less clearly defined, which led us to also apply K-Nearest Neighbors (KNN) Regression and Polynomial Regression. KNN allowed us to explore non-parametric, local patterns in the data,
while Polynomial Regression helped us test whether non-linear relationships might improve prediction accuracy. 



## Results


## Discussion


## Summary



