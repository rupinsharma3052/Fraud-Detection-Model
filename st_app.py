import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header  { font-size:2.5rem; color:#ff4b4b; text-align:center; margin-bottom:1rem; }
    .sub-header   { font-size:1.5rem; color:#2c3e50; margin-top:1rem; margin-bottom:1rem; }
    .fraud-alert  { background:#ff4b4b; color:white; padding:1rem; border-radius:10px; text-align:center; font-weight:bold; }
    .safe-alert   { background:#00cc66; color:white; padding:1rem; border-radius:10px; text-align:center; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    """Cache parsed DataFrame keyed by raw file bytes."""
    import io
    return pd.read_csv(io.BytesIO(file_bytes))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a *copy* of df with engineered columns added."""
    df = df.copy()
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])
    df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balance_diff_dest'] = df['oldbalanceDest'] - df['newbalanceDest']
    df['is_origin_balance_zero'] = (df['oldbalanceOrg'] == 0).astype(int)
    df['is_dest_balance_zero'] = (df['oldbalanceDest'] == 0).astype(int)
    df['amount_to_origin_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['amount_to_dest_ratio'] = df['amount'] / (df['oldbalanceDest'] + 1)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    return df, le


FEATURE_COLS = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest', 'type_encoded',
    'balance_diff_orig', 'balance_diff_dest',
    'is_origin_balance_zero', 'is_dest_balance_zero',
    'amount_to_origin_ratio', 'amount_to_dest_ratio',
]

# ── App shell ──────────────────────────────────────────────────────────────────
def main():
    st.markdown('<div class="main-header">🔍 Financial Fraud Detection System</div>',
                unsafe_allow_html=True)

    st.sidebar.markdown("## 📊 Navigation")
    page = st.sidebar.radio("Select Page", [
        "📁 Data Upload & Overview",
        "📈 Exploratory Data Analysis",
        "🤖 Model Training",
        "🎯 Real-time Prediction",
        "📋 Reports",
    ])

    uploaded_file = st.sidebar.file_uploader(
        "Upload Fraud Dataset (CSV)", type=["csv"],
        help="Upload the fraud analysis dataset in CSV format",
    )

    if uploaded_file is None:
        _show_welcome()
        return

    # Cache by content so re-runs don't re-parse
    df = load_csv(uploaded_file.getvalue())

    required_cols = {'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg',
                     'newbalanceOrig', 'nameDest', 'oldbalanceDest',
                     'newbalanceDest', 'isFraud'}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"Uploaded file is missing columns: {missing}")
        return

    pages = {
        "📁 Data Upload & Overview": data_overview,
        "📈 Exploratory Data Analysis": exploratory_analysis,
        "🤖 Model Training": model_training,
        "🎯 Real-time Prediction": realtime_prediction,
        "📋 Reports": generate_reports,
    }
    pages[page](df)


def _show_welcome():
    st.info("👈 Please upload the Fraud Analysis Dataset CSV file to get started.")
    st.markdown("### Expected Data Format")
    sample = pd.DataFrame({
        'step': [1, 1, 1],
        'type': ['TRANSFER', 'CASH_OUT', 'PAYMENT'],
        'amount': [181.0, 181.0, 9839.64],
        'nameOrig': ['C1305486145', 'C840083671', 'C1231006815'],
        'oldbalanceOrg': [181.0, 181.0, 170136.0],
        'newbalanceOrig': [0.0, 0.0, 160296.36],
        'nameDest': ['C553264065', 'C38997010', 'M1979787155'],
        'oldbalanceDest': [0.0, 21182.0, 0.0],
        'newbalanceDest': [0.0, 0.0, 0.0],
        'isFraud': [1, 1, 0],
    })
    st.dataframe(sample)


# ── Pages ──────────────────────────────────────────────────────────────────────
def data_overview(df: pd.DataFrame):
    st.markdown('<div class="sub-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

    fraud_count = int(df['isFraud'].sum())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("Total Features", df.shape[1])
    col3.metric("Fraudulent Transactions", f"{fraud_count:,}")
    col4.metric("Fraud Percentage", f"{fraud_count / len(df) * 100:.2f}%")

    st.markdown("### 📋 Data Preview (First 100 rows)")
    st.dataframe(df.head(100), use_container_width=True)

    with st.expander("📊 Data Information"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Data Types**")
            st.dataframe(df.dtypes.rename("Data Type").reset_index().rename(columns={"index": "Column"}),
                         use_container_width=True)
        with c2:
            st.markdown("**Missing Values**")
            st.dataframe(df.isnull().sum().rename("Missing Values").reset_index().rename(columns={"index": "Column"}),
                         use_container_width=True)

    with st.expander("📈 Statistical Summary"):
        st.dataframe(df.describe(), use_container_width=True)


def exploratory_analysis(df: pd.DataFrame):
    st.markdown('<div class="sub-header">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Transaction Types", "Amount Analysis",
        "Balance Analysis", "Correlation", "Time Analysis",
    ])

    with tab1:
        type_counts = df['type'].value_counts()
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(x=type_counts.index, y=type_counts.values,
                         title="Transaction Types Distribution",
                         labels={'x': 'Type', 'y': 'Count'},
                         color=type_counts.index)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fbt = df.groupby('type')['isFraud'].sum().reset_index()
            fig = px.bar(fbt, x='type', y='isFraud',
                         title="Fraudulent Transactions by Type",
                         labels={'type': 'Type', 'isFraud': 'Fraud Count'},
                         color='type')
            st.plotly_chart(fig, use_container_width=True)

        fraud_rate = (df.groupby('type')['isFraud']
                      .apply(lambda x: x.mean() * 100)
                      .rename("fraud_rate").reset_index())
        fig = px.bar(fraud_rate, x='type', y='fraud_rate',
                     title="Fraud Rate (%) by Transaction Type",
                     labels={'type': 'Type', 'fraud_rate': 'Fraud Rate (%)'},
                     color='type')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(df, x='type', y='amount', log_y=True,
                         title="Amount Distribution by Type")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(df, x='isFraud', y='amount', log_y=True,
                         title="Amount: Fraud vs Non-Fraud")
            fig.update_xaxes(ticktext=['Non-Fraud', 'Fraud'], tickvals=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Amount Statistics by Fraud Status")
        stats = (df.groupby('isFraud')['amount']
                   .agg(['count', 'mean', 'median', 'std', 'min', 'max'])
                   .round(2))
        stats.index = ['Non-Fraud', 'Fraud']
        st.dataframe(stats, use_container_width=True)

    with tab3:
        sample = df.sample(min(10_000, len(df)), random_state=42)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(sample, x='oldbalanceOrg', y='newbalanceOrig',
                             color='isFraud', opacity=0.5,
                             log_x=True, log_y=True,
                             title="Origin Balance: Before vs After")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(sample, x='oldbalanceDest', y='newbalanceDest',
                             color='isFraud', opacity=0.5,
                             log_x=True, log_y=True,
                             title="Destination Balance: Before vs After")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        num_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                    'oldbalanceDest', 'newbalanceDest', 'isFraud']
        # Sample for correlation to avoid OOM on huge datasets
        corr_sample = df[num_cols].sample(min(50_000, len(df)), random_state=42)
        corr = corr_sample.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                        title="Feature Correlation Matrix",
                        color_continuous_scale='RdBu', zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)

        fraud_corr = (corr['isFraud']
                      .drop('isFraud')
                      .sort_values(ascending=False)
                      .rename("Correlation with Fraud")
                      .reset_index()
                      .rename(columns={"index": "Feature"}))
        fig = px.bar(fraud_corr, x='Feature', y='Correlation with Fraud',
                     color='Correlation with Fraud',
                     color_continuous_scale='RdBu',
                     title="Features Correlated with Fraud")
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        step_counts = df.groupby('step').size().rename("count").reset_index()
        fraud_time = df.groupby('step')['isFraud'].sum().reset_index()

        fig = px.line(step_counts, x='step', y='count',
                      title="Transaction Volume Over Time",
                      labels={'step': 'Time Step', 'count': 'Transactions'})
        st.plotly_chart(fig, use_container_width=True)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=step_counts['step'], y=step_counts['count'],
                                 name="Total", line=dict(color='blue')), secondary_y=False)
        fig.add_trace(go.Scatter(x=fraud_time['step'], y=fraud_time['isFraud'],
                                 name="Fraud", line=dict(color='red')), secondary_y=True)
        fig.update_layout(title="Volume vs Fraud Over Time")
        fig.update_xaxes(title_text="Time Step")
        fig.update_yaxes(title_text="Total Transactions", secondary_y=False)
        fig.update_yaxes(title_text="Fraudulent Transactions", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)


def model_training(df: pd.DataFrame):
    st.markdown('<div class="sub-header">🤖 Machine Learning Model Training</div>',
                unsafe_allow_html=True)

    df_eng, le = engineer_features(df)

    fraud_count = int(df_eng['isFraud'].sum())
    non_fraud_count = len(df_eng) - fraud_count
    st.info(
        f"Class Distribution — Non-Fraud: {non_fraud_count:,} "
        f"({non_fraud_count/len(df_eng)*100:.2f}%)  |  "
        f"Fraud: {fraud_count:,} ({fraud_count/len(df_eng)*100:.2f}%)"
    )

    st.markdown("### Model Configuration")
    c1, c2 = st.columns(2)
    with c1:
        model_choice = st.selectbox("Select Model", ["Random Forest", "Logistic Regression"])
    with c2:
        if model_choice == "Random Forest":
            n_estimators = st.slider("Number of Trees", 50, 300, 100, 50)
            max_depth = st.slider("Max Depth", 5, 30, 10, 5)
        else:
            C_value = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0, 0.01)

    class_weight = st.selectbox("Class Weight Handling", ["balanced", "None"])
    use_balanced = class_weight == "balanced"

    if not st.button("🚀 Train Model", type="primary"):
        return

    X = df_eng[FEATURE_COLS]
    y = df_eng['isFraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    with st.spinner("Training model…"):
        if model_choice == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight='balanced' if use_balanced else None,
                random_state=42, n_jobs=-1,
            )
        else:
            model = LogisticRegression(
                C=C_value,
                class_weight='balanced' if use_balanced else None,
                random_state=42, max_iter=1000,
            )
        model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    # Persist to session state
    st.session_state.update({
        'model': model, 'scaler': scaler,
        'label_encoder': le, 'model_name': model_choice,
    })

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)

    st.markdown("### Model Performance Metrics")
    cols = st.columns(5)
    for col, label, val in zip(cols, ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"],
                                [acc, prec, rec, f1, auc]):
        col.metric(label, f"{val:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Non-Fraud', 'Fraud'], y=['Non-Fraud', 'Fraud'],
                    title="Confusion Matrix", color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)

    # Classification report
    st.markdown("### Detailed Classification Report")
    report = classification_report(y_test, y_pred,
                                   target_names=['Non-Fraud', 'Fraud'],
                                   output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig = go.Figure([
        go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={auc:.4f})'),
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                   line=dict(dash='dash')),
    ])
    fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR",
                      height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance (RF only)
    if model_choice == "Random Forest":
        st.markdown("### Feature Importance")
        fi = (pd.DataFrame({'feature': FEATURE_COLS,
                            'importance': model.feature_importances_})
                .sort_values('importance', ascending=False))
        fig = px.bar(fi.head(15), x='importance', y='feature', orientation='h',
                     title="Top 15 Features", color='importance',
                     color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)


def realtime_prediction(df: pd.DataFrame):
    st.markdown('<div class="sub-header">🎯 Real-time Fraud Prediction</div>',
                unsafe_allow_html=True)

    if 'model' not in st.session_state:
        st.warning("⚠️ Please train a model first in the 'Model Training' tab.")
        return

    st.markdown("### Enter Transaction Details")
    c1, c2 = st.columns(2)
    with c1:
        step            = st.number_input("Time Step", min_value=0, max_value=744, value=1)
        transaction_type = st.selectbox("Transaction Type", sorted(df['type'].unique()))
        amount          = st.number_input("Transaction Amount", min_value=0.0, value=1000.0, step=100.0)
        oldbalanceOrg   = st.number_input("Origin Old Balance", min_value=0.0, value=10000.0, step=100.0)
        newbalanceOrig  = st.number_input("Origin New Balance", min_value=0.0, value=9000.0, step=100.0)
    with c2:
        oldbalanceDest  = st.number_input("Destination Old Balance", min_value=0.0, value=5000.0, step=100.0)
        newbalanceDest  = st.number_input("Destination New Balance", min_value=0.0, value=6000.0, step=100.0)

    if not st.button("🔍 Predict Fraud Risk", type="primary"):
        return

    le: LabelEncoder = st.session_state['label_encoder']
    known_types = list(le.classes_)
    if transaction_type not in known_types:
        st.error(f"Transaction type '{transaction_type}' was not seen during training. "
                 f"Known types: {known_types}")
        return

    type_encoded            = int(le.transform([transaction_type])[0])
    balance_diff_orig       = oldbalanceOrg - newbalanceOrig
    balance_diff_dest       = oldbalanceDest - newbalanceDest
    is_origin_balance_zero  = int(oldbalanceOrg == 0)
    is_dest_balance_zero    = int(oldbalanceDest == 0)
    amount_to_origin_ratio  = amount / (oldbalanceOrg + 1)
    amount_to_dest_ratio    = amount / (oldbalanceDest + 1)

    row = np.array([[
        step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest,
        type_encoded, balance_diff_orig, balance_diff_dest,
        is_origin_balance_zero, is_dest_balance_zero,
        amount_to_origin_ratio, amount_to_dest_ratio,
    ]])
    # Replace any inf values that crept in
    row = np.where(np.isinf(row), 0, row)

    row_scaled = st.session_state['scaler'].transform(row)
    prediction = st.session_state['model'].predict(row_scaled)[0]
    probability = float(st.session_state['model'].predict_proba(row_scaled)[0][1])

    st.markdown("---")
    st.markdown("### Prediction Result")
    if prediction == 1:
        st.markdown(
            f'<div class="fraud-alert">⚠️ HIGH RISK — Fraudulent Transaction Detected! '
            f'(Confidence: {probability:.2%})</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="safe-alert">✅ LOW RISK — Transaction appears legitimate '
            f'(Confidence: {1 - probability:.2%})</div>',
            unsafe_allow_html=True)

    st.markdown("### Risk Score")
    st.progress(min(max(probability, 0.0), 1.0))
    risk_color = "red" if probability > 0.5 else "orange" if probability > 0.3 else "green"
    st.markdown(f'<p style="text-align:center;color:{risk_color};font-size:1.5rem;">'
                f'Risk Score: {probability:.2%}</p>', unsafe_allow_html=True)

    st.markdown("### Risk Factor Analysis")
    risk_factors = []
    if transaction_type in ('TRANSFER', 'CASH_OUT'):
        risk_factors.append("⚠️ High-risk transaction type (TRANSFER / CASH_OUT)")
    if oldbalanceOrg == 0 and amount > 0:
        risk_factors.append("⚠️ Transaction from zero-balance account")
    if newbalanceDest == oldbalanceDest and amount > 0:
        risk_factors.append("⚠️ Destination balance unchanged after transaction")
    if amount > 100_000:
        risk_factors.append("⚠️ Unusually high transaction amount")
    if abs(balance_diff_orig - amount) < 1e-2 and amount > 0:
        risk_factors.append("⚠️ Full balance transfer detected")
    if not risk_factors:
        risk_factors.append("✅ No significant risk factors detected")
    for f in risk_factors:
        st.write(f)


def generate_reports(df: pd.DataFrame):
    st.markdown('<div class="sub-header">📋 Fraud Analysis Reports</div>',
                unsafe_allow_html=True)

    report_type = st.selectbox("Select Report Type", [
        "Fraud Summary Statistics",
        "High-Risk Transaction Types",
        "Top Fraud Amounts",
        "Account Analysis",
    ])

    if report_type == "Fraud Summary Statistics":
        total = len(df)
        fraud = int(df['isFraud'].sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Transactions", f"{total:,}")
        c2.metric("Fraudulent Transactions", f"{fraud:,}")
        c3.metric("Fraud Percentage", f"{fraud/total*100:.2f}%")

        st.markdown("#### By Transaction Type")
        summary = df.groupby('type').agg(
            Fraud_Count=('isFraud', 'sum'),
            Total_Transactions=('isFraud', 'count'),
            Avg_Amount=('amount', 'mean'),
            Total_Amount=('amount', 'sum'),
        ).round(2)
        summary['Fraud_Rate_%'] = (summary['Fraud_Count'] / summary['Total_Transactions'] * 100).round(2)
        st.dataframe(summary, use_container_width=True)

        st.markdown("#### Time-based Summary")
        time_sum = df.groupby('step').agg(
            Fraud_Count=('isFraud', 'sum'),
            Total_Amount=('amount', 'sum'),
        ).reset_index()
        st.dataframe(time_sum, use_container_width=True)

    elif report_type == "High-Risk Transaction Types":
        fraud_by_type = df.groupby('type').apply(
            lambda x: pd.Series({
                'total_transactions': len(x),
                'fraud_count': int(x['isFraud'].sum()),
                'fraud_rate': x['isFraud'].mean() * 100,
                'avg_fraud_amount': x.loc[x['isFraud'] == 1, 'amount'].mean() if x['isFraud'].sum() > 0 else 0,
            })
        ).reset_index()

        fig = px.bar(fraud_by_type, x='type', y='fraud_rate',
                     title="Fraud Rate by Transaction Type", text='fraud_rate',
                     labels={'type': 'Type', 'fraud_rate': 'Fraud Rate (%)'},
                     color='fraud_rate', color_continuous_scale='Reds')
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(fraud_by_type.round(2), use_container_width=True)

    elif report_type == "Top Fraud Amounts":
        top20 = (df[df['isFraud'] == 1]
                   .nlargest(20, 'amount')
                   [['step', 'type', 'amount', 'nameOrig', 'nameDest']])
        st.dataframe(top20, use_container_width=True)

        fig = px.bar(top20, x='amount', y='nameOrig', orientation='h',
                     title="Top 20 Fraudulent Transactions by Amount",
                     labels={'amount': 'Amount', 'nameOrig': 'Origin Account'},
                     color='type', text='amount')
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    elif report_type == "Account Analysis":
        c1, c2 = st.columns(2)
        with c1:
            origins = (df[df['isFraud'] == 1]['nameOrig']
                         .value_counts().head(20)
                         .rename("Fraud Count").reset_index()
                         .rename(columns={"index": "Origin Account"}))
            fig = px.bar(origins, x='Fraud Count', y='nameOrig', orientation='h',
                         title="Top 20 Origin Accounts", color='Fraud Count',
                         color_continuous_scale='Reds',
                         labels={'nameOrig': 'Account'})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            dests = (df[df['isFraud'] == 1]['nameDest']
                       .value_counts().head(20)
                       .rename("Fraud Count").reset_index()
                       .rename(columns={"index": "Destination Account"}))
            fig = px.bar(dests, x='Fraud Count', y='nameDest', orientation='h',
                         title="Top 20 Destination Accounts", color='Fraud Count',
                         color_continuous_scale='Oranges',
                         labels={'nameDest': 'Account'})
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Balance Patterns in Fraudulent Transactions")
        fraud_sample = (df[df['isFraud'] == 1]
                          .sample(min(1000, int(df['isFraud'].sum())), random_state=42))
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Origin Balances", "Destination Balances"))
        fig.add_trace(go.Scatter(x=fraud_sample['oldbalanceOrg'],
                                 y=fraud_sample['newbalanceOrig'],
                                 mode='markers',
                                 marker=dict(size=5, color='red', opacity=0.6),
                                 name='Origin'), row=1, col=1)
        fig.add_trace(go.Scatter(x=fraud_sample['oldbalanceDest'],
                                 y=fraud_sample['newbalanceDest'],
                                 mode='markers',
                                 marker=dict(size=5, color='orange', opacity=0.6),
                                 name='Destination'), row=1, col=2)
        for c in [1, 2]:
            fig.update_xaxes(title_text="Old Balance", type="log", row=1, col=c)
            fig.update_yaxes(title_text="New Balance", type="log", row=1, col=c)
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
