# -*- coding: utf-8 -*-
import pandas as pd
import urllib.request
import zipfile
import os
import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    matthews_corrcoef,
    confusion_matrix
)

# App layout settings
st.set_page_config(page_title="Bank Marketing ML Evaluation Framework", layout="wide")
st.title("🏦 Bank Marketing Dataset Prediction Dashboard")
# ==========================================
# 📍 ADD THE LOGO CODE BLOCK HERE (Line 31)
# ==========================================
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("https://wikimedia.org");
            background-repeat: no-repeat;
            background-size: 380px;
            background-position: center 75%;
            background-attachment: fixed;
        }}
        /* Smooth opacity screen overlay to guarantee high data table and text readability */
        [data-testid="stAppViewContainer"]::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background-color: rgba(255, 255, 255, 0.94);
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# BACKEND COMPONENT: DATA PROCESSING & CACHING
# ==========================================
if not os.path.exists("test_data.csv"):
    with st.spinner("Downloading dataset dependencies for setup..."):
        url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
        urllib.request.urlretrieve(url, "bank_marketing.zip")

        with zipfile.ZipFile("bank_marketing.zip", "r") as zip_ref:
            zip_ref.extractall("bank_data")

        with zipfile.ZipFile("bank_data/bank.zip", "r") as zip_ref_inner:
            zip_ref_inner.extractall(".")

# Load background reference dataset
df = pd.read_csv("bank.csv", sep=";")

# Use robust LabelEncoder logic matching non-numeric data types
le = LabelEncoder()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop('y', axis=1)
y = df['y']

# Establish consistent training/validation split configurations
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Cache training process to prevent slow down when swapping components
@st.cache_resource
def instantiate_and_train_classifiers():
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr_model.fit(X_train_scaled, y_train)

    dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    dt_model.fit(X_train, y_train)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)

    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)

    svm_model = SVC(probability=True, class_weight='balanced', random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    return {
        "Logistic Regression (Balanced)": lr_model, 
        "Decision Tree (Balanced)": dt_model, 
        "K-Nearest Neighbors (k=5)": knn_model, 
        "Naive Bayes": nb_model, 
        "Support Vector Classifier (Balanced)": svm_model
    }

trained_models = instantiate_and_train_classifiers()

# Ensure local test_data.csv exists as a backup file asset
if not os.path.exists("test_data.csv"):
    test_df = X_test.copy()
    test_df["target"] = y_test
    test_df.to_csv("test_data.csv", index=False)

# ==========================================
# REQUIREMENT A: DATASET UPLOAD PORTAL
# ==========================================
st.sidebar.header("📁 Data Source Selection")
uploaded_file = st.sidebar.file_uploader(
    "Upload validation subset (CSV format):", 
    type=["csv"],
    help="Upload your test data matching bank structural features."
)

if uploaded_file is not None:
    eval_dataframe = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom validation set parsed successfully!")
else:
    eval_dataframe = pd.read_csv("test_data.csv")
    st.sidebar.info("Using baseline reference array (`test_data.csv`).")

# Isolate feature vectors from target label metrics if present
if "target" in eval_dataframe.columns:
    X_matrix = eval_dataframe.drop("target", axis=1)
    y_ground_truth = eval_dataframe["target"]
elif "y" in eval_dataframe.columns:
    X_matrix = eval_dataframe.drop("y", axis=1)
    y_ground_truth = eval_dataframe["y"]
else:
    X_matrix = eval_dataframe
    y_ground_truth = None

# Protect against unencoded string vectors uploaded by custom arrays
for col in X_matrix.select_dtypes(exclude=[np.number]).columns:
    X_matrix[col] = le.fit_transform(X_matrix[col].astype(str))

# Create alternative scaled matrix matching target architecture requirements
X_matrix_scaled = scaler.transform(X_matrix)

# ==========================================
# REQUIREMENT B: MODEL SELECTION SELECTBOX
# ==========================================
st.sidebar.header("🤖 Classifier Configuration")
chosen_model_key = st.sidebar.selectbox(
    "Select model variant for target prediction:",
    options=list(trained_models.keys())
)

selected_model = trained_models[chosen_model_key]

# Route raw vs scaled coordinates based on underlying model dependencies
if "Decision Tree" in chosen_model_key:
    evaluation_features = X_matrix
else:
    evaluation_features = X_matrix_scaled

# Execute forward prediction step
y_predictions = selected_model.predict(evaluation_features)

# ==========================================
# DISPLAY INTERFACE PANELS
# ==========================================
with st.expander("🔍 View Active Input Matrix (Top 5 Records)"):
    st.dataframe(eval_dataframe.head(5))

if y_ground_truth is not None:
    # ==========================================
    # REQUIREMENT C: PERFORMANCE SCORE CARD MATRIX
    # ==========================================
    st.markdown(f"### 📈 {chosen_model_key} Performance Metrics")
    
    # Calculate performance indices
    computed_accuracy = accuracy_score(y_ground_truth, y_predictions)
    computed_precision = precision_score(y_ground_truth, y_predictions, zero_division=0)
    computed_recall = recall_score(y_ground_truth, y_predictions, zero_division=0)
    computed_f1 = f1_score(y_ground_truth, y_predictions, zero_division=0)
    computed_mcc = matthews_corrcoef(y_ground_truth, y_predictions)
    
    if hasattr(selected_model, "predict_proba"):
        calculated_probabilities = selected_model.predict_proba(evaluation_features)[:, 1]
        computed_auc = roc_auc_score(y_ground_truth, calculated_probabilities)
    elif hasattr(selected_model, "decision_function"):
        calculated_probabilities = selected_model.decision_function(evaluation_features)
        computed_auc = roc_auc_score(y_ground_truth, calculated_probabilities)
    else:
        computed_auc = roc_auc_score(y_ground_truth, y_predictions)

    # Render metric elements cleanly inside row arrays
    metric_row_1 = st.columns(3)
    metric_row_1[0].metric("Accuracy", f"{computed_accuracy:.4f}")
    metric_row_1[1].metric("Precision (Class 1)", f"{computed_precision:.4f}")
    metric_row_1[2].metric("Recall (Class 1)", f"{computed_recall:.4f}")
    
    metric_row_2 = st.columns(3)
    metric_row_2[0].metric("F1-Score Profile", f"{computed_f1:.4f}")
    metric_row_2[1].metric("ROC AUC Profile", f"{computed_auc:.4f}")
    metric_row_2[2].metric("MCC Coefficient", f"{computed_mcc:.4f}")

    # ==========================================
    # REQUIREMENT D: CONFUSION MATRIX & REPORTS
    # ==========================================
    st.markdown("### 📋 Advanced Statistical Diagnostics")
    
    tab_report, tab_matrix = st.tabs(["📄 Text Classification Report", "🧱 Matrix Heatmap Dataframe"])
    
    with tab_report:
        st.code(classification_report(y_ground_truth, y_predictions), language="text")
        
    with tab_matrix:
        raw_cm = confusion_matrix(y_ground_truth, y_predictions)
        structured_cm_df = pd.DataFrame(
            raw_cm,
            index=["Actual Negative (0)", "Actual Positive (1)"],
            columns=["Predicted Negative (0)", "Predicted Positive (1)"]
        )
        st.dataframe(structured_cm_df, use_container_width=True)
else:
    st.warning("⚠️ No ground-truth label targets tracked inside uploaded structure. Displaying calculated class mapping vector data array:")
    unlabeled_predictions_df = X_matrix.copy()
    unlabeled_predictions_df["Predicted_y_Label"] = y_predictions
    st.dataframe(unlabeled_predictions_df)
