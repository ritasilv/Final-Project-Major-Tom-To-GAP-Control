# Final-Project-Major-Tom-To-GAP-Control


<img src="gap_c.jpg"
     alt="gap"
     style="float: left; margin-right: 15px;" />
     
     

An analysis focused on the Gender Pay Gap of the 2021 Kaggle Machine Learning & Data Science Survey and and a salary predictor based on the responses of the Kagglers.
The gender pay gap or gender wage gap is the average difference between the remuneration for men and women who are working, often doing the same work.
The gender pay gap in the EU stands at 14.1% and has only changed minimally over the last decade. It means that women earn 14.1% on average less per hour than men.
The gender employment gap stood at 11.7% in 2019, with 67.3 % of women across the EU being employed compared to 79% of men. 
(EU27 data)

#### Hypothesis 

Is the gender pay gap real in the Kaggle data set?
Can Random Forest Regressor predict salary in a dataset mostly composed by categorical data?


#### Data Set

Data from 2021 Kaggle Machine Learning & Data Science Survey, with 42+ questions and 25,973 responses 2021. Columns used:

- Age
- Gender
- Country
- Education
- Profession
- Number of programming languages mastered
- Experience in Machine Learning
- Industry
- Size of the company 
- Yearly Salary in USD


#### Data Processing and Data Cleaning

Data was quite clean, nevertheless I have decided to clean the headers, perform some minor cleaning in the columns. Transform the Salary colum which was categorical into numerical and replace the null values of the salary by the average of the column. Dropped answers such profession student and unemployed as my model will focuns on the salary.

Applied boxcox transformation.

####  Tableau

I have created a dashboard with a data distribution page, focused on how the data of the survey is distributed and another one with some analysis concerning the gender pay gap.

#### Models used

For this problem I have decided to use Random Forest Regressor

#### Results and Conclusions

Overall the model did not perform very well, getting a low score not so bad RSME.

Highest score achieved with hyperparamenters: 0.32
RSME: 164.03



#### Main libraries used on the project

- Pandas
- Seaborn
- Matplotlib
- Sklearn


