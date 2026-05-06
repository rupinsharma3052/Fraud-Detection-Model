# fraud_detection_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .fraud-alert {
        background-color: #ff4b4b;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .safe-alert {
        background-color: #00cc66;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    # For the purpose of this app, I'll create the data structure from the CSV content
    # In production, you would load from file:
    # df = pd.read_csv('Fraud_Analysis_Dataset.csv')
    
    # Since we have the data in the conversation, I'll create a sample structure
    # Note: The actual data is large, so I'll create a function to read from the uploaded file
    
    return None  # Placeholder, will use file upload

def load_uploaded_file(uploaded_file):
    """Load the uploaded CSV file"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None

# Main app
def main():
    st.markdown('<div class="main-header">🔍 Financial Fraud Detection System</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 📊 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📁 Data Upload & Overview", "📈 Exploratory Data Analysis", "🤖 Model Training", "🎯 Real-time Prediction", "📋 Reports"]
    )
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload Fraud Dataset (CSV)",
        type=['csv'],
        help="Upload the fraud analysis dataset in CSV format"
    )
    
    if uploaded_file is not None:
        df = load_uploaded_file(uploaded_file)
        
        if page == "📁 Data Upload & Overview":
            data_overview(df)
        elif page == "📈 Exploratory Data Analysis":
            exploratory_analysis(df)
        elif page == "🤖 Model Training":
            model_training(df)
        elif page == "🎯 Real-time Prediction":
            realtime_prediction(df)
        elif page == "📋 Reports":
            generate_reports(df)
    else:
        st.info("👈 Please upload the Fraud Analysis Dataset CSV file to get started")
        
        # Show sample of expected data format
        st.markdown("### Expected Data Format")
        sample_data = {
            'step': [1, 1, 1],
            'type': ['TRANSFER', 'CASH_OUT', 'PAYMENT'],
            'amount': [181, 181, 9839.64],
            'nameOrig': ['C1305486145', 'C840083671', 'C1231006815'],
            'oldbalanceOrg': [181, 181, 170136],
            'newbalanceOrig': [0, 0, 160296.36],
            'nameDest': ['C553264065', 'C38997010', 'M1979787155'],
            'oldbalanceDest': [0, 21182, 0],
            'newbalanceDest': [0, 0, 0],
            'isFraud': [1, 1, 0]
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df)

def data_overview(df):
    st.markdown('<div class="sub-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
    with col2:
        st.metric("Total Features", df.shape[1])
    with col3:
        fraud_count = df['isFraud'].sum()
        st.metric("Fraudulent Transactions", f"{fraud_count:,}")
    with col4:
        fraud_pct = (fraud_count / len(df)) * 100
        st.metric("Fraud Percentage", f"{fraud_pct:.2f}%")
    
    # Dataset preview
    st.markdown("### 📋 Data Preview (First 100 rows)")
    st.dataframe(df.head(100), use_container_width=True)
    
    # Data info
    with st.expander("📊 Data Information"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Data Types**")
            dtype_df = pd.DataFrame(df.dtypes).reset_index()
            dtype_df.columns = ['Column', 'Data Type']
            st.dataframe(dtype_df, use_container_width=True)
        with col2:
            st.markdown("**Missing Values**")
            missing_df = pd.DataFrame(df.isnull().sum()).reset_index()
            missing_df.columns = ['Column', 'Missing Values']
            st.dataframe(missing_df, use_container_width=True)
    
    # Statistical summary
    with st.expander("📈 Statistical Summary"):
        st.dataframe(df.describe(), use_container_width=True)

def exploratory_analysis(df):
    st.markdown('<div class="sub-header">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Transaction Types", "Amount Analysis", "Balance Analysis", "Correlation", "Time Analysis"])
    
    with tab1:
        st.markdown("### Transaction Type Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Transaction type counts
            type_counts = df['type'].value_counts()
            fig = px.bar(
                x=type_counts.index, 
                y=type_counts.values,
                title="Transaction Types Distribution",
                labels={'x': 'Transaction Type', 'y': 'Count'},
                color=type_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Fraud by transaction type
            fraud_by_type = df.groupby('type')['isFraud'].sum().reset_index()
            fig = px.bar(
                fraud_by_type,
                x='type',
                y='isFraud',
                title="Fraudulent Transactions by Type",
                labels={'type': 'Transaction Type', 'isFraud': 'Fraud Count'},
                color='type',
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Fraud rate by transaction type
        st.markdown("### Fraud Rate by Transaction Type")
        fraud_rate = df.groupby('type').apply(lambda x: (x['isFraud'].sum() / len(x)) * 100).reset_index()
        fraud_rate.columns = ['type', 'fraud_rate']
        
        fig = px.bar(
            fraud_rate,
            x='type',
            y='fraud_rate',
            title="Fraud Rate (%) by Transaction Type",
            labels={'type': 'Transaction Type', 'fraud_rate': 'Fraud Rate (%)'},
            color='type',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Transaction Amount Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution of amounts (log scale)
            fig = px.box(
                df,
                x='type',
                y='amount',
                title="Transaction Amount Distribution by Type",
                labels={'type': 'Transaction Type', 'amount': 'Amount'},
                log_y=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Amount comparison: Fraud vs Non-Fraud
            fig = px.box(
                df,
                x='isFraud',
                y='amount',
                title="Amount Distribution: Fraud vs Non-Fraud",
                labels={'isFraud': 'Is Fraud', 'amount': 'Amount'},
                log_y=True
            )
            fig.update_xaxes(ticktext=['Non-Fraud', 'Fraud'], tickvals=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        
        # Amount statistics by fraud status
        st.markdown("### Amount Statistics by Fraud Status")
        amount_stats = df.groupby('isFraud')['amount'].agg(['count', 'mean', 'median', 'std', 'min', 'max']).round(2)
        amount_stats.index = ['Non-Fraud', 'Fraud']
        st.dataframe(amount_stats, use_container_width=True)
    
    with tab3:
        st.markdown("### Balance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Origin account balance changes
            fig = px.scatter(
                df.sample(min(10000, len(df))),
                x='oldbalanceOrg',
                y='newbalanceOrig',
                color='isFraud',
                title="Origin Account Balance: Before vs After",
                labels={'oldbalanceOrg': 'Old Balance', 'newbalanceOrig': 'New Balance'},
                opacity=0.5,
                log_x=True,
                log_y=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Destination account balance changes
            fig = px.scatter(
                df.sample(min(10000, len(df))),
                x='oldbalanceDest',
                y='newbalanceDest',
                color='isFraud',
                title="Destination Account Balance: Before vs After",
                labels={'oldbalanceDest': 'Old Balance', 'newbalanceDest': 'New Balance'},
                opacity=0.5,
                log_x=True,
                log_y=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Balance difference analysis
        st.markdown("### Balance Difference Analysis")
        df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
        df['balance_diff_dest'] = df['oldbalanceDest'] - df['newbalanceDest']
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(
                df,
                x='isFraud',
                y='balance_diff_orig',
                title="Origin Balance Difference: Fraud vs Non-Fraud",
                labels={'isFraud': 'Is Fraud', 'balance_diff_orig': 'Balance Difference'}
            )
            fig.update_xaxes(ticktext=['Non-Fraud', 'Fraud'], tickvals=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                df,
                x='isFraud',
                y='balance_diff_dest',
                title="Destination Balance Difference: Fraud vs Non-Fraud",
                labels={'isFraud': 'Is Fraud', 'balance_diff_dest': 'Balance Difference'}
            )
            fig.update_xaxes(ticktext=['Non-Fraud', 'Fraud'], tickvals=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### Correlation Analysis")
        
        # Select numeric columns for correlation
        numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                       'oldbalanceDest', 'newbalanceDest', 'isFraud']
        correlation_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Key correlations with fraud
        st.markdown("### Key Correlations with Fraud")
        fraud_corr = correlation_matrix['isFraud'].sort_values(ascending=False)
        fraud_corr_df = pd.DataFrame(fraud_corr).reset_index()
        fraud_corr_df.columns = ['Feature', 'Correlation with Fraud']
        
        fig = px.bar(
            fraud_corr_df[fraud_corr_df['Feature'] != 'isFraud'],
            x='Feature',
            y='Correlation with Fraud',
            title="Features Correlated with Fraud",
            color='Correlation with Fraud',
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("### Time-based Analysis")
        
        # Transaction volume over time
        step_counts = df.groupby('step').size().reset_index()
        step_counts.columns = ['step', 'count']
        
        fig = px.line(
            step_counts,
            x='step',
            y='count',
            title="Transaction Volume Over Time",
            labels={'step': 'Time Step', 'count': 'Number of Transactions'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Fraud over time
        fraud_over_time = df.groupby('step')['isFraud'].sum().reset_index()
        
        fig = px.line(
            fraud_over_time,
            x='step',
            y='isFraud',
            title="Fraudulent Transactions Over Time",
            labels={'step': 'Time Step', 'isFraud': 'Number of Fraudulent Transactions'},
            color_discrete_sequence=['red']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Combined view
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=step_counts['step'], y=step_counts['count'], name="Total Transactions", line=dict(color='blue')),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=fraud_over_time['step'], y=fraud_over_time['isFraud'], name="Fraudulent Transactions", line=dict(color='red')),
            secondary_y=True
        )
        
        fig.update_layout(title="Transaction Volume vs Fraud Over Time")
        fig.update_xaxes(title_text="Time Step")
        fig.update_yaxes(title_text="Total Transactions", secondary_y=False)
        fig.update_yaxes(title_text="Fraudulent Transactions", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)

def model_training(df):
    st.markdown('<div class="sub-header">🤖 Machine Learning Model Training</div>', unsafe_allow_html=True)
    
    # Data preprocessing
    st.markdown("### Data Preprocessing")
    
    # Handle transaction types
    le_type = LabelEncoder()
    df['type_encoded'] = le_type.fit_transform(df['type'])
    
    # Select features
    feature_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                    'oldbalanceDest', 'newbalanceDest', 'type_encoded']
    
    # Add engineered features
    df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balance_diff_dest'] = df['oldbalanceDest'] - df['newbalanceDest']
    df['is_origin_balance_zero'] = (df['oldbalanceOrg'] == 0).astype(int)
    df['is_dest_balance_zero'] = (df['oldbalanceDest'] == 0).astype(int)
    df['amount_to_origin_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['amount_to_dest_ratio'] = df['amount'] / (df['oldbalanceDest'] + 1)
    
    feature_cols_extended = feature_cols + ['balance_diff_orig', 'balance_diff_dest', 
                                            'is_origin_balance_zero', 'is_dest_balance_zero',
                                            'amount_to_origin_ratio', 'amount_to_dest_ratio']
    
    # Handle infinite values
    df = df.replace([np.inf, -np.inf], 0)
    
    X = df[feature_cols_extended]
    y = df['isFraud']
    
    # Handle class imbalance
    fraud_count = y.sum()
    non_fraud_count = len(y) - fraud_count
    st.info(f"Class Distribution:\n- Non-Fraud: {non_fraud_count:,} ({non_fraud_count/len(y)*100:.2f}%)\n- Fraud: {fraud_count:,} ({fraud_count/len(y)*100:.2f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model selection
    st.markdown("### Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_choice = st.selectbox(
            "Select Model",
            ["Random Forest", "Logistic Regression"]
        )
    
    with col2:
        if model_choice == "Random Forest":
            n_estimators = st.slider("Number of Trees", 50, 300, 100, 50)
            max_depth = st.slider("Max Depth", 5, 30, 10, 5)
        else:
            C_value = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0, 0.01)
    
    class_weight = st.selectbox(
        "Class Weight Handling",
        ["balanced", "None"]
    )
    
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            if model_choice == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    class_weight='balanced' if class_weight == "balanced" else None,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                model = LogisticRegression(
                    C=C_value,
                    class_weight='balanced' if class_weight == "balanced" else None,
                    random_state=42,
                    max_iter=1000
                )
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Store model in session state
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['features'] = feature_cols_extended
            st.session_state['label_encoder'] = le_type
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Display metrics
            st.markdown("### Model Performance Metrics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
            with col2:
                st.metric("Precision", f"{precision:.4f}")
            with col3:
                st.metric("Recall", f"{recall:.4f}")
            with col4:
                st.metric("F1 Score", f"{f1:.4f}")
            with col5:
                st.metric("AUC-ROC", f"{auc:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Non-Fraud', 'Fraud'],
                y=['Non-Fraud', 'Fraud'],
                title="Confusion Matrix",
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Classification Report
            st.markdown("### Detailed Classification Report")
            report = classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(4), use_container_width=True)
            
            # ROC Curve
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {auc:.4f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
            fig.update_layout(
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                width=600,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance (for Random Forest)
            if model_choice == "Random Forest":
                st.markdown("### Feature Importance")
                feature_importance = pd.DataFrame({
                    'feature': feature_cols_extended,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig = px.bar(
                    feature_importance.head(15),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 15 Most Important Features",
                    color='importance',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)

def realtime_prediction(df):
    st.markdown('<div class="sub-header">🎯 Real-time Fraud Prediction</div>', unsafe_allow_html=True)
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Please train a model first in the 'Model Training' tab")
        return
    
    st.markdown("### Enter Transaction Details for Fraud Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        step = st.number_input("Time Step", min_value=0, max_value=100, value=1)
        transaction_type = st.selectbox("Transaction Type", df['type'].unique())
        amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
        oldbalanceOrg = st.number_input("Origin Account Old Balance", min_value=0.0, value=10000.0)
        newbalanceOrig = st.number_input("Origin Account New Balance", min_value=0.0, value=9000.0)
    
    with col2:
        oldbalanceDest = st.number_input("Destination Account Old Balance", min_value=0.0, value=5000.0)
        newbalanceDest = st.number_input("Destination Account New Balance", min_value=0.0, value=5000.0)
    
    if st.button("🔍 Predict Fraud Risk", type="primary"):
        # Encode transaction type
        type_encoded = st.session_state['label_encoder'].transform([transaction_type])[0]
        
        # Calculate engineered features
        balance_diff_orig = oldbalanceOrg - newbalanceOrig
        balance_diff_dest = oldbalanceDest - newbalanceDest
        is_origin_balance_zero = 1 if oldbalanceOrg == 0 else 0
        is_dest_balance_zero = 1 if oldbalanceDest == 0 else 0
        amount_to_origin_ratio = amount / (oldbalanceOrg + 1)
        amount_to_dest_ratio = amount / (oldbalanceDest + 1)
        
        # Create feature array
        features = [[
            step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest,
            type_encoded, balance_diff_orig, balance_diff_dest, is_origin_balance_zero,
            is_dest_balance_zero, amount_to_origin_ratio, amount_to_dest_ratio
        ]]
        
        # Scale features
        features_scaled = st.session_state['scaler'].transform(features)
        
        # Predict
        prediction = st.session_state['model'].predict(features_scaled)[0]
        probability = st.session_state['model'].predict_proba(features_scaled)[0][1]
        
        # Display result
        st.markdown("---")
        st.markdown("### Prediction Result")
        
        if prediction == 1:
            st.markdown(f'<div class="fraud-alert">⚠️ HIGH RISK - Fraudulent Transaction Detected! (Confidence: {probability:.2%})</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="safe-alert">✅ LOW RISK - Transaction appears legitimate (Confidence: {1-probability:.2%})</div>', unsafe_allow_html=True)
        
        # Risk meter
        st.markdown("### Risk Score")
        risk_color = "red" if probability > 0.5 else "orange" if probability > 0.3 else "green"
        st.progress(float(probability))
        st.markdown(f'<p style="text-align: center; color: {risk_color}; font-size: 1.5rem;">Risk Score: {probability:.2%}</p>', unsafe_allow_html=True)
        
        # Factors contributing to risk
        st.markdown("### Risk Factors Analysis")
        risk_factors = []
        
        if transaction_type in ['TRANSFER', 'CASH_OUT']:
            risk_factors.append("⚠️ High-risk transaction type (TRANSFER/CASH_OUT)")
        
        if oldbalanceOrg == 0 and amount > 0:
            risk_factors.append("⚠️ Transaction from zero-balance account")
        
        if newbalanceDest == oldbalanceDest and amount > 0:
            risk_factors.append("⚠️ Destination balance unchanged after transaction")
        
        if amount > 100000:
            risk_factors.append("⚠️ Unusually high transaction amount")
        
        if balance_diff_orig == amount and amount > 0:
            risk_factors.append("⚠️ Full balance transfer detected")
        
        if not risk_factors:
            risk_factors.append("✅ No significant risk factors detected")
        
        for factor in risk_factors:
            st.write(factor)

def generate_reports(df):
    st.markdown('<div class="sub-header">📋 Fraud Analysis Reports</div>', unsafe_allow_html=True)
    
    report_type = st.selectbox(
        "Select Report Type",
        ["Fraud Summary Statistics", "High-Risk Transaction Types", "Top Fraud Amounts", "Account Analysis"]
    )
    
    if report_type == "Fraud Summary Statistics":
        st.markdown("### Fraud Summary Statistics")
        
        total_transactions = len(df)
        fraud_transactions = df['isFraud'].sum()
        fraud_percentage = (fraud_transactions / total_transactions) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transactions", f"{total_transactions:,}")
        with col2:
            st.metric("Fraudulent Transactions", f"{fraud_transactions:,}")
        with col3:
            st.metric("Fraud Percentage", f"{fraud_percentage:.2f}%")
        
        # Summary by transaction type
        st.markdown("#### Summary by Transaction Type")
        summary_by_type = df.groupby('type').agg({
            'isFraud': ['sum', 'count'],
            'amount': ['mean', 'sum']
        }).round(2)
        summary_by_type.columns = ['Fraud Count', 'Total Transactions', 'Avg Amount', 'Total Amount']
        summary_by_type['Fraud Rate (%)'] = (summary_by_type['Fraud Count'] / summary_by_type['Total Transactions']) * 100
        st.dataframe(summary_by_type, use_container_width=True)
        
        # Time-based summary
        st.markdown("#### Time-based Summary")
        time_summary = df.groupby('step').agg({
            'isFraud': 'sum',
            'amount': 'sum'
        }).reset_index()
        time_summary.columns = ['Time Step', 'Fraud Count', 'Total Amount']
        st.dataframe(time_summary, use_container_width=True)
        
    elif report_type == "High-Risk Transaction Types":
        st.markdown("### High-Risk Transaction Types Analysis")
        
        # Fraud rate by transaction type
        fraud_by_type = df.groupby('type').apply(
            lambda x: pd.Series({
                'total_transactions': len(x),
                'fraud_count': x['isFraud'].sum(),
                'fraud_rate': (x['isFraud'].sum() / len(x)) * 100,
                'avg_fraud_amount': x[x['isFraud'] == 1]['amount'].mean() if x['isFraud'].sum() > 0 else 0
            })
        ).reset_index()
        
        fig = px.bar(
            fraud_by_type,
            x='type',
            y='fraud_rate',
            title="Fraud Rate by Transaction Type",
            text='fraud_rate',
            labels={'type': 'Transaction Type', 'fraud_rate': 'Fraud Rate (%)'},
            color='fraud_rate',
            color_continuous_scale='Reds'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(fraud_by_type, use_container_width=True)
        
    elif report_type == "Top Fraud Amounts":
        st.markdown("### Top 20 Fraudulent Transactions by Amount")
        
        fraud_transactions = df[df['isFraud'] == 1].nlargest(20, 'amount')[['step', 'type', 'amount', 'nameOrig', 'nameDest']]
        st.dataframe(fraud_transactions, use_container_width=True)
        
        fig = px.bar(
            fraud_transactions,
            x='amount',
            y='nameOrig',
            orientation='h',
            title="Top 20 Fraudulent Transactions",
            labels={'amount': 'Transaction Amount', 'nameOrig': 'Origin Account'},
            color='type',
            text='amount'
        )
        fig.update_traces(texttemplate='${:,.0f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
    elif report_type == "Account Analysis":
        st.markdown("### Account Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Origin Accounts Most Involved in Fraud")
            fraud_origins = df[df['isFraud'] == 1]['nameOrig'].value_counts().head(20).reset_index()
            fraud_origins.columns = ['Origin Account', 'Fraud Count']
            
            fig = px.bar(
                fraud_origins,
                x='Fraud Count',
                y='Origin Account',
                orientation='h',
                title="Top 20 Origin Accounts in Fraud",
                color='Fraud Count',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Destination Accounts Most Targeted in Fraud")
            fraud_destinations = df[df['isFraud'] == 1]['nameDest'].value_counts().head(20).reset_index()
            fraud_destinations.columns = ['Destination Account', 'Fraud Count']
            
            fig = px.bar(
                fraud_destinations,
                x='Fraud Count',
                y='Destination Account',
                orientation='h',
                title="Top 20 Destination Accounts in Fraud",
                color='Fraud Count',
                color_continuous_scale='Oranges'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Account balance patterns
        st.markdown("#### Account Balance Patterns in Fraudulent Transactions")
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Origin Balances", "Destination Balances"))
        
        fraud_data = df[df['isFraud'] == 1].sample(min(1000, len(df[df['isFraud'] == 1])))
        
        fig.add_trace(
            go.Scatter(x=fraud_data['oldbalanceOrg'], y=fraud_data['newbalanceOrig'], mode='markers', 
                      marker=dict(size=5, color='red', opacity=0.6), name='Origin'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=fraud_data['oldbalanceDest'], y=fraud_data['newbalanceDest'], mode='markers',
                      marker=dict(size=5, color='orange', opacity=0.6), name='Destination'),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Old Balance", type="log", row=1, col=1)
        fig.update_yaxes(title_text="New Balance", type="log", row=1, col=1)
        fig.update_xaxes(title_text="Old Balance", type="log", row=1, col=2)
        fig.update_yaxes(title_text="New Balance", type="log", row=1, col=2)
        fig.update_layout(height=500, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
