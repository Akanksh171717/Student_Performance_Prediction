🎯 Student Performance Prediction – End-to-End ML Pipeline
📌 Project Overview
This project is an end-to-end machine learning pipeline that predicts student performance scores based on demographic and academic features such as gender, parental education level, lunch type, and test preparation completion.

The workflow includes data analysis, feature engineering, model training, evaluation, and deployment. The pipeline is designed with scalability in mind, allowing easy integration of new datasets and models.

🛠️ Key Features
Exploratory Data Analysis (EDA):

Identified patterns, trends, and correlations in student performance data.

Visualized relationships between socio-economic factors and exam scores.

Data Preprocessing:

Handled missing values, encoded categorical variables, and normalized numerical features.

Automated preprocessing pipeline to ensure consistency for training and prediction.

Model Training & Selection:

Implemented multiple ML algorithms (e.g., CatBoost, Random Forest, Linear Regression).

Tuned hyperparameters for optimal performance.

Performance Metrics:

Evaluated using RMSE, MAE, and R².

Achieved high prediction accuracy with CatBoost as the final model.

Deployment-Ready Pipeline:

Modular src/ structure for data ingestion, transformation, training, and prediction.

Flask API (app.py) for serving predictions.

📂 Project Structure
graphql
Copy
Edit
ML_Projects/
│
├── artifacts/              # Stored trained models, preprocessors, and logs
├── catboost_info/          # CatBoost model training metadata
├── notebook/               # EDA and Problem Statement
├── src/                    # Pipeline scripts (data ingestion, transformation, training)
├── templates/              # HTML templates for web app
│
├── app.py                  # Flask application for model deployment
├── requirements.txt        # Dependencies
├── setup.py                # Package configuration
└── README.md               # Project documentation
📊 Dataset
Source: Custom CSV (stud.csv)
Rows: 1,000 | Columns: 8
Features:

gender – Student gender

race_ethnicity – Group classification

parental_level_of_education – Highest education level of parents

lunch – Type of lunch program

test_preparation_course – Completion status

math_score, reading_score, writing_score – Exam scores

🚀 How to Run
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/Akanksh171717/ML_Projects.git
cd ML_Projects
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the Flask App
bash
Copy
Edit
python app.py
The app will be available at http://127.0.0.1:5000/

📈 Results & Insights
CatBoost outperformed other models with the lowest RMSE and highest R².

Strong positive correlation found between reading and writing scores.

Test preparation course completion increased scores by ~10–15 points.

🛠️ Tech Stack
Languages: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, CatBoost

Frameworks: Flask

Tools: Git, VS Code

📌 Future Improvements
Integrate SHAP for model explainability.

Add support for batch predictions.

Deploy on cloud (Heroku/AWS).

✍️ Author
Akanksh Shetty

LinkedIn

GitHub
