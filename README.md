## 1. Anime recommendation system 
Recommends animes to few users. The source of dataset is [Kaggle Dataset](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database). The Anime recommendation system contains five files: 
- anime_eda containes EDA. 
- a_preprocess does all the processing of data. 
- data_cleaning is a python script to clean data.
- a_model_train, model is trained using KNNwithMeans algorithm.
- a_model_eval, model is trained on KNNwithMean and SVD, compared and top 5 animes are recommended to few users.

## 2. Cyclistic-bike-share
Is a case study to analyze the number of casual riders and members to gain insights on how both users use bikes differently to create proper plans on converting casual riders to members. I have used excel and pandas and added visualizations(charts) using pivot tables in case of excel and seaborn and matplotlib in case of pandas. I have created pivot table for excel of every month, a sample is added in result_images folder along with charts present in jupyter notebook file. This project was done to explore and gain an understanding on excel and to understand concatenation of many large files into a csv file using pandas.I could draw conclusions and few patterns and based on this I was able to recommend 3 solutions which is mentioned in report.

## 3.Credit Card Fraud Detection 
Builds a machine learning–based fraud detection system that identifies fraudulent credit card transactions using anonymized PCA features. It includes an ML pipeline, model evaluation, and a web interface built with Streamlit and Flask. The dataset is based on Kaggle’s Credit Card Fraud Detection Dataset, containing anonymized numerical features `V1–V28`, derived using PCA for confidentiality.
**Features**
-  **Exploratory Data Analysis (EDA)**  
  Correlation plots, outlier analysis, temporal and amount distribution visualization.
- **Class Imbalance Handling**  
  - Undersampling using `RandomUnderSampler`
  - Oversampling using `SMOTE`
  - Cost-sensitive learning (`class_weight='balanced'`)
-  **Machine Learning Models**
  - Random Forest Classifier  
  - Logistic Regression  
  - Support Vector Machine (optional)  
  - Evaluated using Precision-Recall and ROC-AUC curves
- **Deployment**
  - **Flask API** backend for inference  
  - **Streamlit UI** frontend for user input and live prediction
  - Random transaction generation for testing

---
