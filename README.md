# Student Habits vs Academic Performance


| Table of Contents                                          |
| ---------------------------------------------------------- |
| [Introduction](#introduction)                              |
| [Selection of Data](#selection-of-data)                    |
| [Characteristics of data](#characteristics-of-data)         |
| [Methods](#methods)                                        |
| [Materials/APIs/tools used](#materialsapistools-used)       |
| [Results](#results)                                        |
| [Root Mean Squared Error (RMSE) and R-squared value](#root-mean-squared-error-rmse-and-r-squared-value) |
| [Discussion](#discussion)                                  |
| [Key findings](#key-findings)                              |
| [Summary](#summary)                                        |
| [References](#references)                                  |

---
# Introduction
Students usually have busy lives while enrolled in school. Many of their habits can impact their performance in class, especially when taking an exam. In this project, we aim to determine a relationship between a student’s habits and potentially predict their exam score based on those habits. Habits, such as a student's sleep hours, study time, attendance percentage, exercise frequency, and extracurricular participation, will be analyzed. We will be using the [Student Habits vs Academic Performance](https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance) (Nath, 2025) dataset from Kaggle. First, we’ll load and clean the data using pandas (checking for nulls and performing basic sanity checks), then explore relationships using matplotlib (histograms, scatterplots, and correlation). As we complete additional class modules on data wrangling, kNN, and linear regression, we expect to utilize further methods to analyze our data more effectively and predict student performance. We will interpret the results to answer our question about the relationship between the students' habits and their final score performance.

## Selection of Data
We will be using the “Student Habits vs Academic Performance” dataset, which was obtained from Kaggle. It was downloaded on May 25th at 7:30 p.m., and the last update was approximately a month prior to our download. The author of the data is Jayantha Nath. The data is simulated, consisting of 1000 synthetic student records. There are a total of 1000 rows and 16 columns. Each student is assigned a unique student ID, along with their age and gender, to identify them in columns 0, 1, and 2. Columns 3 through 14 represent the habits that students can partake in that can affect their exam scores. These habits range from the hours spent studying to the number of hours spent watching Netflix.  Exam scores are given in the last column. After preprocessing the data, we noticed that there were 91 missing values for the categorical column, `parental_education_level`. These values were replaced with the value `unknown`. We also noticed that outliers existed in the columns `study_hours_per_day` and `social_media_hours`.

### Characteristics of Data:
*   The dataset has a total of 16 columns.
*   Each student is assigned a unique student ID, along with their age and gender.
*   Columns 3 through 14 represent the habits that students can partake in that can affect their exam scores.
*   Exam scores are given in the last column.

## Methods
For this project, we used Python as our programming language, along with the following libraries:
- `pandas` and `numpy` for data manipulation
- `seaborn` and `matplotlib` for data visualization
- `math` for basic mathematical operations
- scikit-learn (`sklearn`) for data modeling, training, and evaluation
- `scipy` for scientific computing

Jupyter Notebook was used by all group members in their local Integrated Development Environments (IDEs). Each group member contributed their own visualizations and analysis of the data, along with their thoughts on the structure and interpretation of the dataset. Responsibilities were divided across the group—for example, one member handled data cleaning, another worked on KNN regression, and others focused on modeling and evaluation. Tasks were assigned and reviewed during our weekly meetings.

After examining the pair plots in the "Understanding the Numerical Data" section and the heatmap in the Correlation Analysis section, we observed several strong linear relationships between student habits and exam scores. This motivated the use of Linear Regression, as it is well-suited for modeling linear trends. However, for some features, the relationships were less clearly defined, which led us to also apply K-Nearest Neighbors (KNN) Regression and Polynomial Regression. KNN allowed us to explore non-parametric, local patterns in the data, while Polynomial Regression helped us test whether non-linear relationships might improve prediction accuracy.

### Materials/APIs/tools used:
*   Python as the programming language
*   Kaggle dataset "Student Habits vs Academic Performance"
*   Jupyter Notebook for data analysis and visualization

## Results
In the three models mentioned above, we compared the Root Mean Squared Error (RMSE) and the R-squared value. The RMSE measures, on average, how far off our predicted exam scores are from the actual scores. For this value, the lower the value, the closer the model is to the actual exam scores. The R-squared value indicated how much of the variation in exam scores can be explained by the model based on student habits. For this value, the higher the value, the stronger the prediction. 
### Root Mean Squared Error (RMSE) and R-squared value:
| Model | RMSE | R-squared |
| --- | --- | --- |
| KNN Regression | 7.37 | - |
| Linear Regression | 5.03 | 0.909 |
| Polynomial Regression | 4.92 | 0.92 |

For the KNN Regression model, we used the top three features: 
- `study_hours_per_day`
- `mental_health_rating`   
- `exercise_frequency`

The model yielded the best RMSE of 7.37 when k = 7. Compared to the blind baseline RMSE of 16.69 we computed, this was a significant improvement.

For the linear regression model, we tested combinations of all numerical columns and used cross-validation to determine which combination of features would produce the best root mean squared error (RMSE). We found that the following seven features would produce the best RMSE:
- `study_hours_per_day`
- `social_media_hours`
- `netflix_hours`
- `attendance_percentage`
- `sleep_hours`
- `exercise_frequency`
- `mental_health_rating`
- `high_score` (column created from binarizing the exam performance at the median)

When evaluated using the model, we found an R-squared value of 0.909 and an RMSE of 5.03, indicating that the model’s predictions are off by approximately 5.03 points on average, and 92% of the variance in exam scores is explained by the model. Compared to the KNN Regression model, this was an improvement.

For the Polynomial Regression, we used the same top features mentioned above with a degree of 2. Our model produced an R-squared value of 0.92 and an RMSE of 4.92. The R-squared value matched that of the KNN model, but the RMSE decreased, thereby reducing the error in the model’s predictions. 

Based on our evaluation metrics, the polynomial regression model was the most accurate, as it had the lowest Root Mean Squared Error (RMSE) and the highest R-squared. Our second-best model would be Linear Regression, followed by KNN Regression. With this model, we can use the intercept and the 44 coefficients to create a function that predicts a student's exam score based on given information about their habits.

## Discussion
Polynomial Regression proved to be the most accurate model for predicting exam scores. This is likely due to its ability to capture nonlinear patterns among features that individually have weak relationships with the target variable, `exam_scores`, with the exception of `study_hours_per_day`, which showed a strong linear correlation. Given the overall weak linear relationships in the dataset, Polynomial Regression was better suited than Linear Regression or KNN Regression for modeling these more complex interactions.

Our results align with the findings in the article [To What Extent Do Study Habits Relate to Performance?](https://pmc.ncbi.nlm.nih.gov/articles/PMC8108503/) (Walck-Shannon et al., 2021), which suggests that three factors influence a student's performance: study strategies, the amount of time spent studying, and the presence of distractions during study sessions. Our research encompassed the latter by using features such as `hours_studying_per_day` and `social_media_hours`, but did not include study strategies. 

With this being said, adding the category of study strategy would be a valuable feature to use as a predictor in future research. The way a student studies is just as beneficial as the amount of time they spend studying. Additionally, it would be beneficial to test our model with a group of students to assess its accuracy. Since we used synthetic data, it would be beneficial to use it as a baseline with actual data from students.  

If given more time or access to a larger dataset, we could have explored additional models, such as Decision Tree Regression, to compare their performance with that of our other models. Decision trees are non-parametric and have better interpretability. It would be valuable to investigate which of our top predictors are used in the nodes of the decision tree. Also, we scaled our data in the KNN and Linear Regression models, but not in the Polynomial Regression model. It would be interesting to see if our model improves when we scale the data.

### Key findings:
*   Polynomial Regression proved to be the most accurate model for predicting exam scores.
*   The results align with the findings in the article "To What Extent Do Study Habits Relate to Performance?" (Walck-Shannon et al., 2021).
*   Adding the category of study strategy would be a valuable feature to use as a predictor in future research.

## Summary
The most important finding from our research is that Polynomial Regression was the most effective model for predicting student exam scores. It outperformed both Linear Regression and KNN Regression, with it having the lowest RMSE (4.92) and the highest R-squared value (0.92). This suggests that nonlinear relationships between study habits and academic performance are significant and better captured by a polynomial model. These results underscore the importance of considering complex patterns, such as interactions and nonlinear trends, when modeling student success.

---
### References:
- Walck-Shannon, E. M., Rowell, S. F., & Frey, R. F. (2021, March). *To what extent do study habits relate to performance?*. CBE life sciences education. [https://pmc.ncbi.nlm.nih.gov/articles/PMC8108503/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8108503/)
- Nath, J. (2025, April 12). *Student habits vs academic performance*. Kaggle. [https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance](https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance) 



