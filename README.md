![Loan Credit Risk](https://www.investopedia.com/thmb/_V-MaR8gF5TPcQYGiapsNoFXhpc=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/creditrisk-Final-18f65d6c12404b9cbccd5bb713b85ce4.jpg)

>source img: https://www.investopedia.com/thmb/_V-MaR8gF5TPcQYGiapsNoFXhpc=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/creditrisk-Final-18f65d6c12404b9cbccd5bb713b85ce4.jpg
# Project Title: loan-credit-risk

## 📝 Overview
As a Data Scientist Intern at ID/X, my job is to create a prediction model using a clien't data from 2007-2014 if a borrower will default their loan or not

## Deployment Link:
https://loan-protection-bramantyo.streamlit.app/

## 🔧 Technologies Used
Programming Language: Python

Libraries: SciPy, Pandas, Seaborn, Matplotlib, NumPy, Joblib, Imbalanced-learn, Scikit-learn, XGBoost, Streamlit

Tools: Jupyter Notebook, Streamlit

## 🚀 Workflow
### 1️ Data Loading
This process involves many steps because I push the data into my local noSQL DB since the data is too big for GitHub. Here are the process:
1. Raw File-->Database
2. Raw Data-->Data Cleaning-->Database (different collection)
3. Clean Data-->Used for EDA & Modelling

### 2️ Exploratory Data Analysis 
Here are the findings that will be used for business insight and Feature Engineering
1. The distribution of loan status is not balanced
2. Most of data are left-skewed
3. There are 4 columns that are autocorrelated (loan_amnt, funded_amnt, funded_amnt_inv, installment)
4. Loan Grade is correlated with Loan Status
5. There are more employed people in the data, but unemployment makes up the majority of the title
6. Employment status (employed or not) has more correlation with loan status than the title itself.
### 3 Feature Engineering
This is the part where I preprocess the data before training them in the model:
1. Feature Selection and Splitting
- In this part, I pick the columns that have correlation with attrition that is shown in the EDA part 9 & 10
- I also split the X and y variable in this part
2. Splitting Train and Test:
- I set the testing size to be 15% of the dataset and the random state is 2
3. Handling Outlier:
- I decide to handle it with MinMax scaler for the column with outlier percentage more than 1%.
4. Pipeline:
- This part is where I preprocess the data so that I can have easier time transforming the data later in both the inference and deployment part
### 4 Model Training, Evaluation, and Tuning
1. Model Training:
- The models that I have trained are 5 in total: Logisti Regression, Decision Tree, Random Forrest, SVC, and XGBoost.
2. Model Evaluation:
- The model that seems to have the best score in our metric of interest is Logistic Regression.
3. Model Tuning:
- I used grid search to find the best parameter for my model
### 5 Model Saving
I saved the model, with the pipeline included using joblib
### 6 Deployment
The deployment uses streamlit where I can test out the result with different types of data. Hopwfully in the future I can create a real app using the model that I trained here.
### 7 Conclusion & Suggestion
The model that is developed has successfully reduce the number of misclassification and MAE, while increasing the Kappa score without risking the possibility of overfitting.

The improvements that this model may need are:
- better null values handling instead of just removing the data
- better outlier handling that are more than just scaling
- More varied data to train the model

## Thank You For Visiting!!

📬 Connect with me and give me feedback

💼 Linkedin: https://www.linkedin.com/in/bramantyo-anandaru-suyadi-0b9729208/ 