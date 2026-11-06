# Exam Performance Prediction Using Machine Learning

A machine learning project that predicts students' examination scores based on academic and lifestyle factors, including study hours, attendance, previous academic performance, and sleep duration.

## Live Demo
https://exam-performance-prediction-bue9kdcvcgafdd4tuyhxjm.streamlit.app/

## Project Overview

Student academic performance can be influenced by several factors. This project explores how machine learning can be used to understand these factors and predict students' examination performance.

The project uses **Random Forest Regression**, a supervised machine learning algorithm, to estimate examination scores based on historical student data.

The main objective is to develop a predictive model that can help identify academic performance patterns and support data-driven educational decisions.

## Project Objectives

- Analyze student academic performance data.
- Clean and preprocess inconsistent datasets.
- Explore relationships between student habits and examination scores.
- Train a machine learning regression model.
- Evaluate model performance using Mean Absolute Error (MAE).
- Predict examination scores for new student records.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and cleaning |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Scikit-learn | Machine learning |
| Jupyter Notebook | Model development and experimentation |

## Project Structure

```text
exam-performance-prediction/
│
├── examprediction.ipynb
├── student_performance_messy.csv
├── README.md
├── requirements.txt
└── .gitignore
```

The `.venv/` directory is local and should not be uploaded to GitHub.

## Dataset Description

The project uses a student performance dataset containing academic and lifestyle information.

### Features

| Feature | Description |
|---|---|
| study_hours | Number of hours spent studying |
| attendance_percent | Student attendance percentage |
| previous_score | Student's previous academic score |
| sleep_hours | Number of hours spent sleeping |

### Target Variable

- `exam_score` – The examination score the model aims to predict.

## Data Preprocessing

The dataset undergoes several cleaning operations before model training:

1. Removing duplicate records.
2. Standardizing numerical values.
3. Removing units such as `hrs` and `%`.
4. Converting inconsistent text values into numerical values.
5. Handling invalid and out-of-range values.
6. Identifying missing values.
7. Filling missing feature values using training-set medians.

## Machine Learning Model

This project uses **Random Forest Regression** from Scikit-learn.

The model learns patterns between student characteristics and examination scores.

### Model Training

The dataset is divided into:

- 80% training data.
- 20% testing data.

The model uses the following input features:

```python
X = df[
    [
        "study_hours",
        "attendance_percent",
        "previous_score",
        "sleep_hours"
    ]
]

y = df["exam_score"]
```

### Model Evaluation

The model is evaluated using Mean Absolute Error (MAE).

```python
from sklearn.metrics import mean_absolute_error

predictions = model.predict(x_test)

mae = mean_absolute_error(y_test, predictions)

print("MAE:", round(mae, 1))
```

MAE measures the average absolute difference between actual examination scores and predicted scores. Lower values indicate better predictive performance.

## Getting Started

Follow these instructions to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/exam-performance-prediction.git
```

### 2. Navigate into the Project

```bash
cd exam-performance-prediction
```

### 3. Create a Virtual Environment

```bash
python3 -m venv .venv
```

### 4. Activate the Environment

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Jupyter Notebook

```bash
jupyter notebook
```

Open `examprediction.ipynb` and execute the cells.

## Example Prediction

The notebook demonstrates how a new student's information can be passed to the trained model.

```python
new_student = pd.DataFrame({
    "study_hours": [6],
    "attendance_percent": [90],
    "previous_score": [75],
    "sleep_hours": [8]
})

score = model.predict(new_student)[0]

print("Predicted score:", round(score))
```

The model returns an estimated examination score based on patterns learned during training.

## Future Improvements

Planned improvements include:

- Building an interactive web application using Streamlit.
- Comparing Random Forest with Linear Regression and other algorithms.
- Improving feature engineering.
- Adding more student performance indicators.
- Developing visual dashboards for academic performance analysis.
- Saving and loading trained models for deployment.
- Providing personalized academic improvement recommendations.

## Disclaimer

This project is developed for educational and machine learning experimentation purposes. Predictions are estimates and should not be treated as guaranteed examination outcomes or the sole basis for academic decisions.

## Author

**Dennis Mutiso**

Software Engineering Student | Aspiring Machine Learning Engineer

Interested in software engineering, machine learning, data science, and building technology-driven solutions.

## License

This project is intended for educational and learning purposes. A formal open-source license can be added as the project develops.
