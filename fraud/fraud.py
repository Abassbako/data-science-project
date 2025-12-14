# Fraud.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
import shap

# PAGE CONFIG
st.set_page_config(
    page_title="Advanced Fraud Detection Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SIDEBAR – APP NAVIGATION
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Data Explorer", "Model Performance", "Predictions", "Explainability"]
)

# UPLOAD DATA
st.sidebar.header("📤 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

df = load_data(uploaded_file)

# TARGET SELECTION
st.sidebar.header("🎯 Target Selection")
TARGET = st.sidebar.selectbox("Select target column", df.columns)

X = df.drop(columns=[TARGET])
y = df[TARGET]

# BINARY TARGET CHECK / CONVERSION
if y.nunique() > 2:
    st.warning("Target has more than 2 unique values. Converting to binary automatically.")
    # Example: mark any positive value as fraud (1), else legitimate (0)
    y = (y > 0).astype(int)
    
# AUTO-TRAIN MODEL
@st.cache_resource
def train_model(X, y):
    y_bin = y.astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost classifier
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        objective='binary:logistic'
    )
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_test_scaled, y_test

model, scaler, X_test_scaled, y_test = train_model(X, y)

# OVERVIEW PAGE
if page == "Overview":
    st.title("📊 Fraud Detection Dashboard")
    st.markdown(
        """
        ### Business Objective
        Detect fraudulent transactions using machine learning to reduce financial loss and risk exposure.

        ### Key Capabilities
        - Interactive data exploration
        - Real-time predictions
        - Model evaluation & explainability
        - Production-style deployment structure
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(df))
    col2.metric("Fraud Cases", int(y.sum()))
    col3.metric("Fraud Rate (%)", round(y.mean() * 100, 2))

# DATA EXPLORER
elif page == "Data Explorer":
    st.title("📂 Data Explorer")

    # Dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head(100))

    # Summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(df.describe().T)  # Transpose for readability

    # Feature distribution visualization
    st.subheader("Feature Distribution")
    feature = st.selectbox("Select feature to visualize", X.columns)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[feature], bins=30, color="skyblue", edgecolor="black")
    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Correlation heatmap
    if st.checkbox("Show Correlation Heatmap"):
        st.subheader("Feature Correlation Heatmap")
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(corr, cmap="coolwarm")
        plt.colorbar(cax)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)
        st.pyplot(fig)

# MODEL PERFORMANCE
elif page == "Model Performance":
    st.title("📈 Model Performance")

    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Classification Report")
        report = classification_report(y_test, preds, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        plt.colorbar(cax)
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, val, ha='center', va='center')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    st.subheader("ROC Curve")
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0, 1], [0, 1], linestyle="--")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()
    st.pyplot(fig2)

# PREDICTIONS
elif page == "Predictions":
    st.title("🔮 Make a Prediction")
    input_data = {}
    st.markdown("### Enter Feature Values")
    for col in X.columns:
        input_data[col] = st.number_input(f"{col}", value=float(X[col].mean()))

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if pred == 1:
            st.error(f"⚠️ Fraud Detected (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Legitimate Transaction (Probability: {1 - prob:.2f})")

# EXPLAINABILITY
elif page == "Explainability":
    st.title("🧠 Model Explainability")
    st.markdown("This section explains **why** the model makes predictions.")
    
    sample_size = st.slider(
    "Sample size",
    min_value=50,
    max_value=max(300, len(X)),
    value=min(100, len(X))
)
    X_sample = X.sample(sample_size, random_state=42)
    X_sample_scaled = scaler.transform(X_sample)

    explainer = shap.Explainer(model, X_sample_scaled)
    shap_values = explainer(X_sample_scaled)

    st.subheader("SHAP Beeswarm Plot")
    shap.summary_plot(shap_values, features=X_sample_scaled, feature_names=X.columns, show=False)
    st.pyplot(plt.gcf())

# FOOTER
st.markdown(
    """
    ---
    **Built with Streamlit | Machine Learning | Explainable AI**
    """
)
