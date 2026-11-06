
import re
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Exam Performance Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("Exam Performance Prediction")
st.write(
    "A machine learning application that predicts "
    "student examination performance using academic "
    "and lifestyle factors."
)


# ==========================================
# 2. CONFIGURATION
# ==========================================

DATASET = "student_performance_messy.csv"

FEATURES = [
    "study_hours",
    "attendance_percent",
    "previous_score",
    "sleep_hours"
]

TARGET = "exam_score"

LIMITS = {
    "study_hours": (0, 24),
    "attendance_percent": (0, 100),
    "previous_score": (0, 100),
    "sleep_hours": (0, 24),
    "exam_score": (0, 100)
}

WORD_NUMBERS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100"
}


# ==========================================
# 3. NUMERICAL DATA CLEANING
# ==========================================

def clean_numeric(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    if value in ["", "nan", "none", "null", "na", "n/a"]:
        return np.nan

    # Convert written numbers
    if value in WORD_NUMBERS:
        value = WORD_NUMBERS[value]

    # Extract the first numerical value.
    # Examples:
    # "55 / 100" -> 55
    # "81%" -> 81
    # "1.5 hrs" -> 1.5
    # "88.0" -> 88

    match = re.search(r"-?\d+(?:\.\d+)?", value)

    if match:
        try:
            return float(match.group())
        except ValueError:
            return np.nan

    return np.nan


# ==========================================
# 4. LOAD AND CLEAN DATASET
# ==========================================

@st.cache_data
def load_dataset():

    df = pd.read_csv(DATASET)

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # Remove duplicate records
    original_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = original_rows - len(df)

    # Check required columns
    required_columns = FEATURES + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Clean numerical columns
    for col in required_columns:
        df[col] = df[col].apply(clean_numeric)

    # Replace impossible values with missing values
    for col, (minimum, maximum) in LIMITS.items():

        invalid = (
            (df[col] < minimum) |
            (df[col] > maximum)
        )

        df.loc[invalid, col] = np.nan

    # Target cannot be missing
    rows_before_target = len(df)

    df = df.dropna(subset=[TARGET])

    invalid_targets_removed = (
        rows_before_target - len(df)
    )

    # Ensure at least one usable value exists per feature
    empty_features = [
        col for col in FEATURES
        if df[col].notna().sum() == 0
    ]

    if empty_features:
        raise ValueError(
            f"These features contain no valid numerical data: "
            f"{empty_features}. Check your CSV values."
        )

    statistics = {
        "original_rows": original_rows,
        "clean_rows": len(df),
        "duplicates_removed": duplicates_removed,
        "invalid_targets_removed": invalid_targets_removed
    }

    return df, statistics


# ==========================================
# 5. TRAIN MACHINE LEARNING MODEL
# ==========================================

@st.cache_resource
def train_model():

    df, statistics = load_dataset()

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    if len(df) < 5:
        raise ValueError(
            "At least 5 valid records are required "
            "to train the model."
        )

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Preprocessing and machine learning pipeline
    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = (
        r2_score(y_test, predictions)
        if len(y_test) >= 2
        else np.nan
    )

    return (
        model,
        df,
        statistics,
        mae,
        r2,
        len(X_train),
        len(X_test)
    )


# ==========================================
# 6. LOAD APPLICATION DATA
# ==========================================

try:

    (
        model,
        df,
        statistics,
        mae,
        r2,
        train_size,
        test_size
    ) = train_model()

except Exception as error:

    st.error(f"Application error: {error}")

    st.info(
        "Check your CSV file, column names and numerical values."
    )

    st.stop()


# ==========================================
# 7. SIDEBAR
# ==========================================

st.sidebar.title("About This Project")

st.sidebar.write(
    "Exam Performance Prediction uses "
    "Random Forest Regression to estimate "
    "student examination scores."
)

st.sidebar.divider()

st.sidebar.write("Developed by Dennis Mutiso")

st.sidebar.write(
    "Technologies: Python, Pandas, "
    "Scikit-learn and Streamlit."
)


# ==========================================
# 8. DATASET DIAGNOSTICS
# ==========================================

with st.expander("Dataset Diagnostics", expanded=False):

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Original Records",
        statistics["original_rows"]
    )

    col2.metric(
        "Clean Records",
        statistics["clean_rows"]
    )

    col3.metric(
        "Duplicates Removed",
        statistics["duplicates_removed"]
    )

    st.subheader("Missing Values After Cleaning")

    st.dataframe(
        df[FEATURES + [TARGET]].isnull().sum()
        .rename("Missing Values")
        .to_frame(),
        use_container_width=True
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Data Cleaning Summary")

    st.write(
        "Invalid target records removed:",
        statistics["invalid_targets_removed"]
    )


# ==========================================
# 9. MODEL PERFORMANCE
# ==========================================

st.divider()

st.header("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Mean Absolute Error",
    f"{mae:.2f}"
)

col2.metric(
    "R² Score",
    f"{r2:.3f}" if np.isfinite(r2) else "N/A"
)

col3.metric(
    "Training Records",
    train_size
)

col4.metric(
    "Testing Records",
    test_size
)

st.caption(
    "MAE represents the average absolute prediction error. "
    "R² measures how well the model explains score variation."
)


# ==========================================
# 10. STUDENT PREDICTION INTERFACE
# ==========================================

st.divider()

st.header("Predict Student Examination Performance")

st.write(
    "Enter the student's academic and lifestyle "
    "information below."
)

col1, col2 = st.columns(2)

with col1:

    study_hours = st.number_input(
        "Study Hours Per Day",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    attendance = st.number_input(
        "Attendance Percentage",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=1.0
    )

with col2:

    previous_score = st.number_input(
        "Previous Examination Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )

    sleep_hours = st.number_input(
        "Sleep Hours Per Day",
        min_value=0.0,
        max_value=24.0,
        value=8.0,
        step=0.5
    )


# ==========================================
# 11. MAKE PREDICTION
# ==========================================

if st.button(
    "Predict Examination Score",
    type="primary",
    use_container_width=True
):

    student = pd.DataFrame(
        [[
            study_hours,
            attendance,
            previous_score,
            sleep_hours
        ]],
        columns=FEATURES
    )

    try:

        prediction = model.predict(student)[0]

        prediction = float(
            np.clip(prediction, 0, 100)
        )

        st.divider()

        st.header("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Examination Score",
                f"{prediction:.1f} / 100"
            )

        with col2:

            st.metric(
                "Model MAE",
                f"{mae:.2f}"
            )

        st.progress(
            int(round(prediction))
        )

        if prediction >= 70:

            st.success(
                "Estimated performance category: High"
            )

        elif prediction >= 50:

            st.info(
                "Estimated performance category: Moderate"
            )

        else:

            st.warning(
                "Estimated performance category: Needs Improvement"
            )

        st.subheader("Student Information")

        st.dataframe(
            student,
            use_container_width=True
        )

        st.caption(
            "This prediction is an estimate based on "
            "patterns learned from the training dataset. "
            "It is not a guaranteed examination result."
        )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "Exam Performance Prediction | "
    "Developed by Dennis Mutiso | "
    "Python and Machine Learning"
)
