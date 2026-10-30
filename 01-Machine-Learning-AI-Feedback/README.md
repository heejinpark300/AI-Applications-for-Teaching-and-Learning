
# Project 01 | Machine Learning-Based AI Feedback Acceptance Predictor

## Educational Application
Predicting English learners' AI feedback acceptance, revision, and rejection behaviors.

## AI Technologies Implemented
- Python for data generation and preprocessing
- Scikit-learn for machine learning
- Logistic Regression and Random Forest
- Accuracy and Macro F1-score evaluation
- Confusion Matrix and Feature Importance
- Streamlit for interactive web application development
- GitHub and Streamlit Community Cloud for deployment

## Project Overview
This project demonstrates an end-to-end machine learning workflow, from dataset preparation and model evaluation to interactive web application development and deployment.

The prototype classifies AI feedback responses into three categories:
- Accept
- Edit
- Reject

## Dataset
- 300 synthetic records
- Three input features: AI trust, English proficiency, and AI experience
- One target variable: feedback acceptance behavior
- Training/test split: 80/20

## Results

| Model | Accuracy | Macro F1-score |
|---|---:|---:|
| Logistic Regression | 0.3500 | 0.2684 |
| Random Forest | 0.3000 | 0.2978 |

Random Forest was used for the interactive prototype.

## Research Significance
This prototype establishes a technical foundation for future behavioral studies of AI trust, feedback literacy, and trust calibration in English language education.

The current dataset is randomly generated for demonstration and does not represent empirical learner behavior.

## Source Code
The deployed application source code is currently maintained in the repository root as `app.py`.

## Deployment
The application has been deployed using Streamlit Community Cloud.
