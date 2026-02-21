####################################################################################################
# Imports
####################################################################################################
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


####################################################################################################
# Streamlit Page Config
####################################################################################################
st.set_page_config(page_title=" Loan Approval Predictor", layout="wide")

st.title(" Loan Approval Prediction")
st.caption("Machine Learning classification project using Logistic Regression (practice purpose).")


####################################################################################################
# Data Loading (cached)
####################################################################################################
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


####################################################################################################
# Model Training (cached)
####################################################################################################
@st.cache_resource
def train_model(df: pd.DataFrame):

    target_col = "approved"

    drop_cols = [target_col]
    if "applicant_name" in df.columns:
        drop_cols.append("applicant_name")

    X = df.drop(columns=drop_cols)
    y = df[target_col]

    categorical_cols = [c for c in ["gender", "city", "employment_type", "bank"] if c in X.columns]
    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    # Pipelines
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return model, metrics, X_train.columns.tolist()


####################################################################################################
# Sidebar – Dataset
####################################################################################################
st.sidebar.header("1️⃣ Load Dataset")

csv_path = st.sidebar.text_input(
    "CSV file path",
    value="loan_dataset.csv",
    help="Keep as-is if CSV is in the same folder"
)

try:
    df = load_data(csv_path)
    st.sidebar.success(f"Loaded {len(df):,} rows")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()


####################################################################################################
# Sidebar – Train Model
####################################################################################################
st.sidebar.header("2️⃣ Train Model")

if st.sidebar.button("Train / Re-train Model"):
    st.cache_resource.clear()

model, metrics, feature_order = train_model(df)


####################################################################################################
# Main Layout
####################################################################################################
left, right = st.columns(2)

with left:
    st.subheader(" Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

with right:
    st.subheader(" Model Performance (Test Set)")

    st.write({
        "Accuracy": round(metrics["accuracy"], 4),
        "Precision": round(metrics["precision"], 4),
        "Recall": round(metrics["recall"], 4),
        "F1 Score": round(metrics["f1"], 4),
    })

    cm = metrics["confusion_matrix"]
    st.write("Confusion Matrix (rows = actual, columns = predicted)")
    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        ),
        use_container_width=True
    )

st.divider()


####################################################################################################
# Prediction UI
####################################################################################################
st.subheader(" Try a Prediction")

c1, c2, c3, c4 = st.columns(4)

with c1:
    applicant_name = st.text_input("Applicant Name", "Muhammad Ali")
    gender = st.selectbox("Gender", ["M", "F"])
    age = st.slider("Age", 21, 60, 30)

with c2:
    city = st.selectbox("City", sorted(df["city"].unique()))
    employment_type = st.selectbox("Employment Type", sorted(df["employment_type"].unique()))
    bank = st.selectbox("Bank", sorted(df["bank"].unique()))

with c3:
    monthly_income_pkr = st.number_input("Monthly Income (PKR)", 1500, 500000, 120000, step=1000)
    credit_score = st.slider("Credit Score", 300, 900, 680)

with c4:
    loan_amount_pkr = st.number_input("Loan Amount (PKR)", 50000, 3500000, 800000, step=5000)
    loan_tenure_months = st.selectbox("Loan Tenure (Months)", [6, 12, 18, 24, 36, 48, 60])
    existing_loans = st.selectbox("Existing Loans", [0, 1, 2, 3])
    default_history = st.selectbox("Default History", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    has_credit_card = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")


####################################################################################################
# Build Input Row
####################################################################################################
input_df = pd.DataFrame([{
    "gender": gender,
    "age": age,
    "city": city,
    "employment_type": employment_type,
    "bank": bank,
    "monthly_income_pkr": monthly_income_pkr,
    "credit_score": credit_score,
    "loan_amount_pkr": loan_amount_pkr,
    "loan_tenure_months": loan_tenure_months,
    "existing_loans": existing_loans,
    "default_history": default_history,
    "has_credit_card": has_credit_card
}])

input_df = input_df[feature_order]


####################################################################################################
# Prediction
####################################################################################################
if st.button("Predict Approval"):
    prob = model.predict_proba(input_df)[0, 1]
    prediction = int(prob >= 0.5)

    if prediction == 1:
        st.success(f" {applicant_name}: APPROVED (Probability: {prob:.2%})")
    else:
        st.error(f" {applicant_name}: REJECTED (Probability: {prob:.2%})")





