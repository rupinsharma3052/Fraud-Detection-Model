import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import warnings
import time
import base64

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .prediction-fraud {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        border-left: 5px solid #d32f2f;
        animation: pulse 2s infinite;
    }
    .prediction-legit {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        border-left: 5px solid #388e3c;
    }
    .fraud-warning {
        background-color: #d32f2f;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .risk-high {
        background-color: #d32f2f;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #ff9800;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-low {
        background-color: #4caf50;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(211, 47, 47, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(211, 47, 47, 0); }
        100% { box-shadow: 0 0 0 0 rgba(211, 47, 47, 0); }
    }
    .stButton > button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
    }
    .status-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🔍 Financial Fraud Detection System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'encoder' not in st.session_state:
    st.session_state.encoder = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'results' not in st.session_state:
    st.session_state.results = None

# Function to load and preprocess data
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None

# Function to engineer features
def engineer_features(df):
    df = df.copy()
    
    # Create balance difference features
    df['balanceOrgDiff'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['balanceDestDiff'] = df['newbalanceDest'] - df['oldbalanceDest']
    
    # Create ratio features
    df['orgBalanceRatio'] = df['newbalanceOrig'] / (df['oldbalanceOrg'] + 1)
    df['destBalanceRatio'] = df['newbalanceDest'] / (df['oldbalanceDest'] + 1)
    
    # Create zero balance flags
    df['isOrgBalanceZero'] = (df['oldbalanceOrg'] == 0).astype(int)
    df['isDestBalanceZero'] = (df['oldbalanceDest'] == 0).astype(int)
    
    # Create amount ratio features
    df['amountToOrgRatio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['amountToDestRatio'] = df['amount'] / (df['oldbalanceDest'] + 1)
    
    return df

# Function to prepare data for modeling
def prepare_data(df):
    # Engineer features
    df = engineer_features(df)
    
    # One-hot encode transaction type
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    type_encoded = encoder.fit_transform(df[['type']])
    type_encoded_df = pd.DataFrame(
        type_encoded,
        columns=['type_' + cat for cat in encoder.categories_[0][1:]]
    )
    
    # Prepare features and target
    X = df.drop(['isFraud', 'nameOrig', 'nameDest', 'type'], axis=1)
    X = pd.concat([X.reset_index(drop=True), type_encoded_df.reset_index(drop=True)], axis=1)
    y = df['isFraud']
    
    return X, y, encoder

# Function to train models
def train_models(X_train, y_train, X_test, y_test):
    results = {}
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(probability=True, random_state=42)
    }
    
    for name, model in models.items():
        with st.spinner(f'Training {name}...'):
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            y_pred = model.predict(X_test)
            
            # Get probabilities if available
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_prob)
            except:
                y_prob = y_pred
                roc_auc = None
            
            results[name] = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc,
                'train_time': train_time,
                'y_pred': y_pred,
                'y_prob': y_prob if roc_auc is not None else None
            }
    
    return results

# Function to analyze transaction for fraud indicators
def analyze_transaction(transaction_data):
    """Analyze transaction for potential fraud indicators"""
    indicators = []
    risk_score = 0
    
    # Check 1: Unusual amount
    if transaction_data['amount'] > 1000000:
        indicators.append("⚠️ Extremely high transaction amount (> $1,000,000)")
        risk_score += 30
    elif transaction_data['amount'] > 100000:
        indicators.append("⚠️ Very high transaction amount (> $100,000)")
        risk_score += 20
    elif transaction_data['amount'] > 50000:
        indicators.append("⚠️ High transaction amount (> $50,000)")
        risk_score += 10
    
    # Check 2: Zero balance before transaction
    if transaction_data['oldbalanceOrg'] == 0:
        indicators.append("⚠️ Origin account had zero balance before transaction")
        risk_score += 15
    
    # Check 3: Complete balance depletion
    if transaction_data['newbalanceOrig'] == 0 and transaction_data['oldbalanceOrg'] > 0:
        indicators.append("⚠️ Origin account completely depleted after transaction")
        risk_score += 25
    
    # Check 4: Negative balance (impossible - data error)
    if transaction_data['newbalanceOrig'] < 0 or transaction_data['newbalanceDest'] < 0:
        indicators.append("❌ Negative balance detected - data inconsistency")
        risk_score += 40
    
    # Check 5: Large percentage transfer
    if transaction_data['oldbalanceOrg'] > 0:
        percent_transferred = (transaction_data['amount'] / transaction_data['oldbalanceOrg']) * 100
        if percent_transferred > 90:
            indicators.append(f"⚠️ Transferred {percent_transferred:.1f}% of origin balance")
            risk_score += 20
        elif percent_transferred > 50:
            indicators.append(f"⚠️ Transferred {percent_transferred:.1f}% of origin balance")
            risk_score += 10
    
    # Check 6: Suspicious transaction types
    if transaction_data['type'] in ['TRANSFER', 'CASH_OUT']:
        if transaction_data['amount'] > 50000:
            indicators.append("⚠️ High-risk transaction type (TRANSFER/CASH_OUT) with large amount")
            risk_score += 15
    
    # Check 7: Destination zero balance
    if transaction_data['oldbalanceDest'] == 0 and transaction_data['type'] in ['TRANSFER', 'CASH_OUT']:
        indicators.append("⚠️ Destination account newly created or inactive")
        risk_score += 10
    
    return indicators, min(risk_score, 100)

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Data Upload")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success(f"✅ Data loaded successfully! ({len(df)} rows)")
            
            # Show data preview
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head())
            
            with st.expander("📈 Data Info"):
                st.write(f"**Shape:** {df.shape}")
                st.write(f"**Columns:** {list(df.columns)}")
                st.write(f"**Fraud Rate:** {df['isFraud'].mean()*100:.2f}%")
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.25)
    smote_enabled = st.checkbox("Apply SMOTE (Balance Classes)", value=True)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This system detects fraudulent financial transactions using multiple Machine Learning models.
    
    **Fraud Indicators:**
    - Unusual transaction amounts
    - Account balance anomalies
    - Suspicious transaction patterns
    - High-risk transaction types
    """)

# Main content area
if not st.session_state.data_loaded:
    st.info("👈 Please upload a CSV file to get started!")
    st.markdown("""
    ### Expected CSV format:
    The file should contain the following columns:
    - `step` - Time step of transaction
    - `type` - Transaction type (PAYMENT, TRANSFER, CASH_OUT, etc.)
    - `amount` - Transaction amount
    - `nameOrig` - Origin account name
    - `oldbalanceOrg` - Origin account balance before transaction
    - `newbalanceOrig` - Origin account balance after transaction
    - `nameDest` - Destination account name
    - `oldbalanceDest` - Destination account balance before transaction
    - `newbalanceDest` - Destination account balance after transaction
    - `isFraud` - Target variable (1 = Fraud, 0 = Legitimate)
    """)
else:
    # Data Analysis Section
    st.markdown("## 📊 Exploratory Data Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{len(st.session_state.df):,}")
    with col2:
        fraud_count = st.session_state.df['isFraud'].sum()
        fraud_rate = fraud_count / len(st.session_state.df) * 100
        st.metric("Fraudulent Transactions", f"{fraud_count:,}", delta=f"{fraud_rate:.2f}%")
    with col3:
        legit_count = len(st.session_state.df) - fraud_count
        st.metric("Legitimate Transactions", f"{legit_count:,}")
    with col4:
        avg_amount = st.session_state.df['amount'].mean()
        st.metric("Average Amount", f"${avg_amount:,.2f}")
    
    # Charts
    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "📈 Correlations", "📉 Transaction Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            fraud_counts = st.session_state.df['isFraud'].value_counts()
            colors = ['#4CAF50', '#f44336']
            bars = ax.bar(['Legitimate', 'Fraud'], fraud_counts.values, color=colors)
            ax.set_title('Fraud Distribution')
            ax.set_ylabel('Count')
            for bar, count in zip(bars, fraud_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                        f'{count:,}', ha='center', va='bottom')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            type_counts = st.session_state.df['type'].value_counts()
            colors = plt.cm.Set3(range(len(type_counts)))
            ax.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', colors=colors)
            ax.set_title('Transaction Types Distribution')
            st.pyplot(fig)
        
        # Transaction amount distribution
        fig, ax = plt.subplots(figsize=(10, 4))
        st.session_state.df[st.session_state.df['isFraud'] == 0]['amount'].hist(bins=50, alpha=0.5, label='Legitimate', ax=ax)
        st.session_state.df[st.session_state.df['isFraud'] == 1]['amount'].hist(bins=50, alpha=0.5, label='Fraud', ax=ax)
        ax.set_xlabel('Amount')
        ax.set_ylabel('Frequency')
        ax.set_title('Amount Distribution by Transaction Type')
        ax.legend()
        ax.set_yscale('log')
        st.pyplot(fig)
    
    with tab2:
        # Correlation heatmap
        numeric_cols = st.session_state.df.select_dtypes(include=['int64', 'float64']).columns
        correlation = st.session_state.df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
        ax.set_title('Correlation Heatmap')
        st.pyplot(fig)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x='isFraud', y='amount', data=st.session_state.df, ax=ax)
            ax.set_title('Amount by Fraud Status')
            ax.set_xticklabels(['Legitimate', 'Fraud'])
            ax.set_yscale('log')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x='type', hue='isFraud', data=st.session_state.df, ax=ax)
            ax.set_title('Fraud by Transaction Type')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
    
    # Model Training Section
    st.markdown("---")
    st.markdown("## 🤖 Model Training & Evaluation")
    
    if st.button("🚀 Train Models", use_container_width=True):
        with st.spinner("Preparing data..."):
            # Prepare data
            X, y, encoder = prepare_data(st.session_state.df)
            st.session_state.encoder = encoder
            st.session_state.feature_columns = X.columns.tolist()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            st.session_state.scaler = scaler
            
            # Apply SMOTE if enabled
            if smote_enabled:
                with st.spinner("Applying SMOTE..."):
                    smote = SMOTE(random_state=42)
                    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
                    st.success(f"SMOTE applied - Training set balanced: {y_train_balanced.sum():.0f} fraud samples")
            else:
                X_train_balanced, y_train_balanced = X_train_scaled, y_train
            
            st.success("Data preparation complete!")
        
        # Train models
        with st.spinner("Training models (this may take a few minutes)..."):
            results = train_models(X_train_balanced, y_train_balanced, X_test_scaled, y_test)
            st.session_state.results = results
            st.session_state.model_trained = True
            st.session_state.X_test = X_test_scaled
            st.session_state.y_test = y_test
        
        # Find best model
        best_name = max(results, key=lambda x: results[x]['f1'])
        st.session_state.best_model_name = best_name
        st.session_state.best_model = results[best_name]['model']
        
        st.success(f"✅ Training complete! Best model: {best_name} (F1 Score: {results[best_name]['f1']:.4f})")
    
    # Display Results
    if st.session_state.model_trained and st.session_state.results is not None:
        st.markdown("---")
        st.markdown("## 📈 Model Performance Results")
        
        # Create results dataframe
        results_df = pd.DataFrame([
            {
                'Model': name,
                'Accuracy': st.session_state.results[name]['accuracy'],
                'Precision': st.session_state.results[name]['precision'],
                'Recall': st.session_state.results[name]['recall'],
                'F1 Score': st.session_state.results[name]['f1'],
                'ROC AUC': st.session_state.results[name]['roc_auc'] if st.session_state.results[name]['roc_auc'] is not None else 'N/A',
                'Train Time (s)': st.session_state.results[name]['train_time']
            }
            for name in st.session_state.results
        ]).sort_values('F1 Score', ascending=False)
        
        st.dataframe(results_df.style.format({
            'Accuracy': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1 Score': '{:.4f}',
            'ROC AUC': '{:.4f}',
            'Train Time (s)': '{:.2f}'
        }), use_container_width=True)
        
        # Performance Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            for metric in metrics:
                values = results_df[metric].values[:5]
                ax.bar([f"{results_df['Model'].iloc[i][:15]}\n{metric}" for i in range(len(values))], 
                       values, alpha=0.7)
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison (Top 5)')
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
        
        with col2:
            # Best model confusion matrix
            best_result = st.session_state.results[st.session_state.best_model_name]
            cm = confusion_matrix(st.session_state.y_test, best_result['y_pred'])
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Legitimate', 'Fraud'],
                       yticklabels=['Legitimate', 'Fraud'])
            ax.set_title(f'Confusion Matrix - {st.session_state.best_model_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
        
        # ROC Curves
        fig, ax = plt.subplots(figsize=(8, 6))
        for name in st.session_state.results:
            if st.session_state.results[name]['roc_auc'] is not None:
                fpr, tpr, _ = roc_curve(st.session_state.y_test, st.session_state.results[name]['y_prob'])
                ax.plot(fpr, tpr, label=f"{name} (AUC = {st.session_state.results[name]['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(loc='lower right')
        st.pyplot(fig)
        
        # Feature Importance (if available)
        if hasattr(st.session_state.best_model, 'feature_importances_'):
            st.markdown("---")
            st.markdown("## 🔍 Feature Importance Analysis")
            
            feature_importance = pd.DataFrame({
                'Feature': st.session_state.feature_columns,
                'Importance': st.session_state.best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15), ax=ax)
            ax.set_title(f'Top 15 Feature Importances - {st.session_state.best_model_name}')
            st.pyplot(fig)
    
    # Prediction Section (Always show if model is trained)
    if st.session_state.model_trained and st.session_state.best_model is not None:
        st.markdown("---")
        st.markdown("## 🔮 Live Fraud Detection")
        st.markdown("Enter transaction details to check if it's fraudulent:")
        
        with st.expander("📝 Transaction Details", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                transaction_type = st.selectbox(
                    "Transaction Type",
                    ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT'],
                    help="Type of transaction being performed",
                    key="transaction_type_select"
                )
                amount = st.number_input(
                    "Transaction Amount ($)",
                    min_value=0.0,
                    value=1000.0,
                    step=100.0,
                    help="Amount being transferred",
                    key="amount_input"
                )
            
            with col2:
                oldbalanceOrg = st.number_input(
                    "Origin Balance Before ($)",
                    min_value=0.0,
                    value=5000.0,
                    step=500.0,
                    help="Origin account balance before transaction",
                    key="oldbalanceOrg_input"
                )
                newbalanceOrig = st.number_input(
                    "Origin Balance After ($)",
                    min_value=0.0,
                    value=4000.0,
                    step=500.0,
                    help="Origin account balance after transaction",
                    key="newbalanceOrig_input"
                )
            
            with col3:
                oldbalanceDest = st.number_input(
                    "Destination Balance Before ($)",
                    min_value=0.0,
                    value=10000.0,
                    step=500.0,
                    help="Destination account balance before transaction",
                    key="oldbalanceDest_input"
                )
                newbalanceDest = st.number_input(
                    "Destination Balance After ($)",
                    min_value=0.0,
                    value=11000.0,
                    step=500.0,
                    help="Destination account balance after transaction",
                    key="newbalanceDest_input"
                )
        
        if st.button("🔍 Analyze Transaction", use_container_width=True, type="primary", key="analyze_button"):
            # Create feature vector
            transaction_data = {
                'step': 1,
                'amount': amount,
                'oldbalanceOrg': oldbalanceOrg,
                'newbalanceOrig': newbalanceOrig,
                'oldbalanceDest': oldbalanceDest,
                'newbalanceDest': newbalanceDest,
                'type': transaction_type
            }
            
            # Analyze for fraud indicators
            indicators, risk_score = analyze_transaction(transaction_data)
            
            input_data = {
                'step': [1],
                'amount': [amount],
                'oldbalanceOrg': [oldbalanceOrg],
                'newbalanceOrig': [newbalanceOrig],
                'oldbalanceDest': [oldbalanceDest],
                'newbalanceDest': [newbalanceDest],
                'balanceOrgDiff': [newbalanceOrig - oldbalanceOrg],
                'balanceDestDiff': [newbalanceDest - oldbalanceDest],
                'orgBalanceRatio': [newbalanceOrig / (oldbalanceOrg + 1)],
                'destBalanceRatio': [newbalanceDest / (oldbalanceDest + 1)],
                'isOrgBalanceZero': [1 if oldbalanceOrg == 0 else 0],
                'isDestBalanceZero': [1 if oldbalanceDest == 0 else 0],
                'amountToOrgRatio': [amount / (oldbalanceOrg + 1)],
                'amountToDestRatio': [amount / (oldbalanceDest + 1)]
            }
            
            # Add one-hot encoded type
            for cat in ['CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']:
                input_data[f'type_{cat}'] = [1 if transaction_type == cat else 0]
            
            input_df = pd.DataFrame(input_data)
            
            # Ensure columns match training
            for col in st.session_state.feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            input_df = input_df[st.session_state.feature_columns]
            
            # Scale
            input_scaled = st.session_state.scaler.transform(input_df)
            
            # Predict
            prediction = st.session_state.best_model.predict(input_scaled)[0]
            
            try:
                fraud_probability = st.session_state.best_model.predict_proba(input_scaled)[0][1]
            except:
                fraud_probability = None
            
            # Save to history
            st.session_state.last_prediction = {
                'fraud': bool(prediction),
                'probability': fraud_probability,
                'transaction_type': transaction_type,
                'amount': amount,
                'risk_score': risk_score,
                'indicators': indicators
            }
            st.session_state.prediction_history.append(st.session_state.last_prediction)
            
            # Display result
            if prediction == 1:
                st.markdown("""
                <div class="prediction-fraud">
                    <div class="status-icon">⚠️</div>
                    <h2 style="color: #d32f2f; margin: 0;">FRAUD DETECTED</h2>
                    <p style="font-size: 1.2rem; margin-top: 0.5rem;">This transaction has been flagged as potentially fraudulent</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk score and probability
                col1, col2, col3 = st.columns(3)
                with col1:
                    if fraud_probability:
                        st.metric("Fraud Probability", f"{fraud_probability:.2%}", delta="High Risk")
                with col2:
                    risk_color = "risk-high" if risk_score > 70 else "risk-medium" if risk_score > 30 else "risk-low"
                    st.markdown(f'<div class="{risk_color}" style="text-align:center; padding:0.5rem; border-radius:5px;"><strong>Risk Score: {risk_score}/100</strong></div>', unsafe_allow_html=True)
                with col3:
                    st.metric("Transaction Type", transaction_type)
            else:
                st.markdown("""
                <div class="prediction-legit">
                    <div class="status-icon">✅</div>
                    <h2 style="color: #388e3c; margin: 0;">LEGITIMATE TRANSACTION</h2>
                    <p style="font-size: 1.2rem; margin-top: 0.5rem;">This transaction appears legitimate</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if fraud_probability:
                        st.metric("Fraud Probability", f"{fraud_probability:.2%}", delta="Low Risk")
                with col2:
                    risk_color = "risk-high" if risk_score > 70 else "risk-medium" if risk_score > 30 else "risk-low"
                    st.markdown(f'<div class="{risk_color}" style="text-align:center; padding:0.5rem; border-radius:5px;"><strong>Risk Score: {risk_score}/100</strong></div>', unsafe_allow_html=True)
                with col3:
                    st.metric("Transaction Type", transaction_type)
            
            # Display fraud indicators
            if indicators:
                st.markdown("### 🚨 Fraud Indicators Detected:")
                for indicator in indicators:
                    st.warning(indicator)
            else:
                st.info("✅ No suspicious indicators detected in this transaction")
            
            # Transaction Summary
            st.markdown("### 📋 Transaction Summary")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.markdown(f"""
                **Origin Account:**
                - Balance Before: ${oldbalanceOrg:,.2f}
                - Balance After: ${newbalanceOrig:,.2f}
                - Change: ${newbalanceOrig - oldbalanceOrg:,.2f}
                """)
            
            with summary_col2:
                st.markdown(f"""
                **Destination Account:**
                - Balance Before: ${oldbalanceDest:,.2f}
                - Balance After: ${newbalanceDest:,.2f}
                - Change: ${newbalanceDest - oldbalanceDest:,.2f}
                """)
            
            with summary_col3:
                st.markdown(f"""
                **Transaction Details:**
                - Type: {transaction_type}
                - Amount: ${amount:,.2f}
                - Status: {'⚠️ FRAUD' if prediction == 1 else '✅ LEGITIMATE'}
                """)
        
        # Prediction History
        if st.session_state.prediction_history:
            st.markdown("---")
            st.markdown("## 📜 Recent Predictions")
            
            history_df = pd.DataFrame([
                {
                    'Date': f"Prediction {i+1}",
                    'Amount': f"${p['amount']:,.2f}",
                    'Type': p['transaction_type'],
                    'Status': '⚠️ FRAUD' if p['fraud'] else '✅ LEGITIMATE',
                    'Risk Score': p['risk_score'],
                    'Fraud Prob': f"{p['probability']:.2%}" if p['probability'] else "N/A"
                }
                for i, p in enumerate(st.session_state.prediction_history[-10:])
            ])
            
            st.dataframe(history_df, use_container_width=True)
    
    # Export results
    if st.session_state.model_trained and st.session_state.results is not None:
        st.markdown("---")
        st.markdown("## 📥 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export predictions
            results_summary = pd.DataFrame({
                'Model': list(st.session_state.results.keys()),
                'F1 Score': [st.session_state.results[m]['f1'] for m in st.session_state.results],
                'Precision': [st.session_state.results[m]['precision'] for m in st.session_state.results],
                'Recall': [st.session_state.results[m]['recall'] for m in st.session_state.results]
            })
            
            csv = results_summary.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="model_results.csv">📊 Download Results CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            if st.session_state.prediction_history:
                history_export = pd.DataFrame([
                    {
                        'Amount': p['amount'],
                        'Transaction Type': p['transaction_type'],
                        'Is_Fraud': p['fraud'],
                        'Risk_Score': p['risk_score'],
                        'Fraud_Probability': p['probability']
                    }
                    for p in st.session_state.prediction_history
                ])
                history_csv = history_export.to_csv(index=False)
                b64 = base64.b64encode(history_csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="prediction_history.csv">📜 Download Prediction History</a>'
                st.markdown(href, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 👨‍💻 Fraud Detection System")
st.markdown("Built with ❤️ using Streamlit and Scikit-learn")
