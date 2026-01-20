🎯 Netflix Content Type Prediction (Machine Learning Project)
📌 Project Overview

This project builds a Machine Learning classification model to predict whether a Netflix title is a Movie or a TV Show based on selected features.
The goal is to understand the end-to-end ML workflow, from data preprocessing to model evaluation.

🧠 Problem Statement

Netflix hosts thousands of titles across different formats.
The objective of this project is to classify Netflix content as either Movie or TV Show using historical metadata.

This project focuses on:

Feature engineering

Binary classification

Model evaluation

Understanding ML limitations

📊 Dataset

Source: Netflix Movies and TV Shows dataset

File: netflix_titles.csv

Records: ~8,800

Type: Structured CSV data

Columns Used
Column	Description
release_year	Year the content was released
duration	Duration of content (minutes or seasons)
type	Movie or TV Show (target variable)
⚙️ Feature Engineering

Extracted numeric values from duration

Converted duration to numerical format

Handled missing values using median imputation

df['duration_num'] = df['duration'].str.extract(r'(\d+)').astype(float)
df['duration_num'] = df['duration_num'].fillna(df['duration_num'].median())

🎯 Target Variable

Column: type

Encoding:

Movie → 0

TV Show → 1

Encoded using LabelEncoder.

🤖 Model Used
Logistic Regression

Reason for selection:

Suitable for binary classification

Interpretable and fast

Strong baseline model for ML projects

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

🏋️ Model Training

Train/Test Split: 80% / 20%

Random State: 42 (for reproducibility)

train_test_split(test_size=0.2, random_state=42)

📈 Model Evaluation
Metrics Used

Accuracy

Confusion Matrix

accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)

Result

Accuracy: ~70–75% (may vary)

Performance is reasonable given limited features

⚠️ Limitations

Very limited feature set

Duration mixes minutes and seasons

No text-based features (title, description)

Logistic Regression assumes linear patterns

Class imbalance not explicitly handled

🔮 Future Improvements

Separate logic for Movies vs TV Shows duration

Add country, rating, and cast features

Apply text features using NLP

Try advanced models (Random Forest, XGBoost)

Handle class imbalance properly

🛠️ Tools & Technologies

Python

Pandas

Scikit-learn

Google Colab

GitHub

📁 Project Structure
Data-Analytics-Portfolio/
│
├── Week5_Netflix_ML.ipynb
├── netflix_titles.csv
├── README.md

✅ Key Learning Outcomes

End-to-end ML pipeline understanding

Importance of feature engineering

Handling missing values in ML

Model evaluation beyond accuracy

Real-world ML limitations

👤 Author

Thirumala Naidu
Aspiring Data Analyst transitioning towards AI & Machine Learning