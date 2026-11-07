"""
Hotel Booking Demand Prediction & Model Evaluation Dashboard
BITS Pilani - Work Integrated Learning Programmes Division
M.Tech (AIML) - Machine Learning Assignment 2
Author: Gandikota Kameswara Sastry
"""

import os
os.environ['MPLCONFIGDIR'] = os.environ.get('TEMP', '/tmp')
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# Set page layout and metadata
st.set_page_config(
    page_title="Hotel Booking Demand Classifier | BITS Pilani ML Assignment 2",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished academic UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0d3b66 0%, #001e3d 100%);
        padding: 24px;
        border-radius: 10px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
    }
    .main-header p {
        margin: 6px 0 0 0;
        font-size: 14px;
        color: #e0e1dd;
    }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 14px 18px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .metric-title {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        color: #6c757d;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #0d3b66;
        margin-top: 4px;
    }
    .metric-desc {
        font-size: 11px;
        color: #adb5bd;
    }
    .winner-badge {
        background-color: #2a9d8f;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Helper & Cached Functions
# ==============================================================================
@st.cache_resource
def load_models_and_artifacts():
    """Loads saved models, preprocessor, and feature configuration."""
    model_dir = 'model'
    preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.joblib'))
    
    with open(os.path.join(model_dir, 'feature_columns.json'), 'r') as f:
        feature_config = json.load(f)

    model_files = {
        'Logistic Regression': 'logistic_regression.joblib',
        'Decision Tree': 'decision_tree.joblib',
        'KNN': 'knn.joblib',
        'Naive Bayes': 'naive_bayes.joblib',
        'Support Vector Machine': 'svm.joblib'
    }

    models = {}
    for name, filename in model_files.items():
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            loaded_model = joblib.load(path)
            # Scikit-learn cross-version compatibility guard (e.g., multi_class attribute in LogisticRegression)
            if not hasattr(loaded_model, 'multi_class') and hasattr(loaded_model, 'classes_'):
                setattr(loaded_model, 'multi_class', 'deprecated')
            models[name] = loaded_model
        else:
            st.error(f"Missing artifact: {path}")

    return preprocessor, feature_config, models


def clean_and_prepare_test_df(df_input: pd.DataFrame, feature_config: dict):
    """
    Standardizes raw input dataframe to match model preprocessing schema.
    """
    df = df_input.copy()

    # Impute missing values if present
    if 'country' in df.columns:
        df['country'] = df['country'].fillna('Unknown')
    if 'agent' in df.columns:
        df['agent'] = df['agent'].fillna(0).astype(int)
    if 'company' in df.columns:
        df['company'] = df['company'].fillna(0).astype(int)
    if 'children' in df.columns:
        df['children'] = df['children'].fillna(0).astype(int)

    # Feature engineering if raw columns exist
    if 'stays_in_weekend_nights' in df.columns and 'stays_in_week_nights' in df.columns:
        if 'total_stay_nights' not in df.columns:
            df['total_stay_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

    if 'adults' in df.columns and 'children' in df.columns and 'babies' in df.columns:
        if 'total_guests' not in df.columns:
            df['total_guests'] = df['adults'] + df['children'] + df['babies']
        if 'is_family' not in df.columns:
            df['is_family'] = ((df['children'] > 0) | (df['babies'] > 0)).astype(int)

    if 'reserved_room_type' in df.columns and 'assigned_room_type' in df.columns:
        if 'room_type_changed' not in df.columns:
            df['room_type_changed'] = (df['reserved_room_type'] != df['assigned_room_type']).astype(int)

    if 'country' in df.columns and 'country_group' not in df.columns:
        top_countries = ['PRT', 'GBR', 'FRA', 'ESP', 'DEU', 'ITA', 'IRL', 'BEL', 'BRA', 'NLD']
        df['country_group'] = df['country'].apply(lambda c: c if c in top_countries else 'Other')

    # Drop leakage columns if present
    drop_cols = ['reservation_status', 'reservation_status_date', 'arrival_date_year', 'assigned_room_type', 'country']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def calculate_metrics_dict(y_true, y_pred, y_prob):
    """Calculates all 6 mandatory metrics specified by the assignment."""
    acc = accuracy_score(y_true, y_pred)
    try:
        auc_val = roc_auc_score(y_true, y_prob)
    except Exception:
        auc_val = 0.5
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        'Accuracy': acc,
        'AUC': auc_val,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'MCC': mcc
    }


# ==============================================================================
# Main Streamlit Application
# ==============================================================================
def main():
    # Header Banner
    st.markdown("""
    <div class="main-header">
        <h1>🏨 Hotel Booking Demand — Predictive Classifier & Evaluation</h1>
        <p><strong>BITS Pilani — Work Integrated Learning Programmes (WILP) | M.Tech (AIML) Machine Learning Assignment 2</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # Load artifacts
    try:
        preprocessor, feature_config, models = load_models_and_artifacts()
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}. Please ensure `model/` directory exists with all .joblib files.")
        st.stop()

    required_features = feature_config['all_features']

    # --------------------------------------------------------------------------
    # Sidebar Controls
    # --------------------------------------------------------------------------
    st.sidebar.header("📁 1. Dataset Upload (Test Data)")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Test Data CSV (Must contain input features and `is_canceled` column)",
        type=['csv'],
        help="Upload only test data as Streamlit free tier has limited RAM."
    )

    # Load default test_data.csv if no user upload
    default_test_path = 'test_data.csv'
    if uploaded_file is not None:
        try:
            test_df = pd.read_csv(uploaded_file)
            data_source_label = f"Uploaded File ({uploaded_file.name})"
            st.sidebar.success(f"Loaded {len(test_df):,} records from uploaded CSV.")
        except Exception as e:
            st.sidebar.error(f"Failed to read CSV: {e}")
            st.stop()
    else:
        if os.path.exists(default_test_path):
            test_df = pd.read_csv(default_test_path)
            data_source_label = f"Default Benchmark Test Data ({default_test_path})"
            st.sidebar.info(f"Using default `{default_test_path}` ({len(test_df):,} records).")
        else:
            st.error("No test data found. Please upload a test dataset CSV.")
            st.stop()

    # Provide sample download button
    if os.path.exists(default_test_path):
        with open(default_test_path, 'rb') as f:
            st.sidebar.download_button(
                label="⬇️ Download Sample Test CSV",
                data=f,
                file_name="sample_hotel_test_data.csv",
                mime="text/csv",
                help="Download sample formatted test CSV to test file upload functionality."
            )

    st.sidebar.markdown("---")
    st.sidebar.header("🤖 2. Model Selection")
    model_options = [
        "All Models (Comparative Benchmark)",
        "Decision Tree",
        "Logistic Regression",
        "KNN",
        "Naive Bayes",
        "Support Vector Machine"
    ]
    selected_model_name = st.sidebar.selectbox(
        "Select Classification Model to Evaluate:",
        model_options,
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ 3. Classification Settings")
    prob_threshold = st.sidebar.slider(
        "Decision Threshold (Probability of Cancellation)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Default is 0.50. Adjusting changes the precision/recall trade-off."
    )

    st.sidebar.markdown("""
    ---
    **Submission Metadata:**
    - Course: Machine Learning (Assignment 2)
    - Institution: BITS Pilani (WILP)
    - Evaluation: 5 Models × 6 Metrics
    """)

    # --------------------------------------------------------------------------
    # Verify Schema & Preprocess Test Data
    # --------------------------------------------------------------------------
    if 'is_canceled' not in test_df.columns:
        st.error("Uploaded test CSV must contain the target column `is_canceled` for evaluation.")
        st.stop()

    # Clean and standardize test dataframe
    prepared_df = clean_and_prepare_test_df(test_df, feature_config)

    # Check for missing feature columns
    missing_cols = [col for col in required_features if col not in prepared_df.columns]
    if missing_cols:
        st.error(f"The uploaded dataset is missing required features: {missing_cols[:5]} (and others). Please use sample test format.")
        st.stop()

    X_test_raw = prepared_df[required_features]
    y_test = prepared_df['is_canceled'].astype(int)

    # Transform using saved preprocessor
    try:
        X_test_proc = preprocessor.transform(X_test_raw)
    except Exception as e:
        st.error(f"Error transforming features with saved preprocessor: {e}")
        st.stop()

    # Data Summary Bar
    with st.expander("🔍 Dataset Overview & Target Class Distribution", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active Test Source", data_source_label.split('(')[0].strip())
        c2.metric("Total Test Instances", f"{len(test_df):,}")
        c3.metric("Confirmed Bookings (0)", f"{(y_test == 0).sum():,} ({((y_test == 0).mean()*100):.1f}%)")
        c4.metric("Canceled Bookings (1)", f"{(y_test == 1).sum():,} ({((y_test == 1).mean()*100):.1f}%)")
        st.dataframe(test_df.head(5), use_container_width=True)

    # --------------------------------------------------------------------------
    # Mode A: All Models (Comparative Benchmark)
    # --------------------------------------------------------------------------
    if selected_model_name == "All Models (Comparative Benchmark)":
        st.subheader("📊 Comparative Evaluation of All 5 Machine Learning Models")
        st.write(f"Evaluated on **{len(y_test):,} test instances** using saved trained models (no retraining).")

        all_metrics = []
        roc_data = {}

        for name, model in models.items():
            if not hasattr(model, 'multi_class') and hasattr(model, 'classes_'):
                setattr(model, 'multi_class', 'deprecated')
            if hasattr(model, 'predict_proba'):
                try:
                    y_prob = model.predict_proba(X_test_proc)[:, 1]
                except AttributeError:
                    setattr(model, 'multi_class', 'deprecated')
                    y_prob = model.predict_proba(X_test_proc)[:, 1]
            elif hasattr(model, 'decision_function'):
                df_val = model.decision_function(X_test_proc)
                y_prob = 1 / (1 + np.exp(-df_val))
            else:
                y_prob = model.predict(X_test_proc)

            y_pred = (y_prob >= prob_threshold).astype(int)
            m_dict = calculate_metrics_dict(y_test, y_pred, y_prob)
            m_dict['ML Model Name'] = name
            all_metrics.append(m_dict)
            roc_data[name] = y_prob

        results_df = pd.DataFrame(all_metrics)[
            ['ML Model Name', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        ]

        # Display table with formatting
        st.dataframe(
            results_df.style.format({
                'Accuracy': '{:.4f}',
                'AUC': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1': '{:.4f}',
                'MCC': '{:.4f}'
            }).highlight_max(
                subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                color='#d4edda'
            ),
            use_container_width=True
        )

        st.caption("Green highlights denote the best performing model for each respective metric.")

        # Visual Comparison Tabs
        tab1, tab2, tab3 = st.tabs(["📈 ROC Curves", "📊 Metrics Comparison Bar Chart", "📝 Performance Observations"])

        with tab1:
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = {
                'Decision Tree': '#2ca02c',
                'Logistic Regression': '#1f77b4',
                'Support Vector Machine': '#ff7f0e',
                'KNN': '#9467bd',
                'Naive Bayes': '#d62728'
            }
            for name, y_prob in roc_data.items():
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_score = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_score:.4f})", color=colors.get(name, '#333333'), lw=2)

            ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Chance Line (0.50)")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
            ax.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=11, fontweight='bold')
            ax.set_title('Receiver Operating Characteristic (ROC) Comparison', fontsize=12, fontweight='bold')
            ax.legend(loc="lower right")
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

        with tab2:
            fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
            plot_df = results_df.melt(
                id_vars=['ML Model Name'],
                value_vars=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                var_name='Metric', value_name='Score'
            )
            sns.barplot(data=plot_df, x='Metric', y='Score', hue='ML Model Name', palette='tab10', ax=ax_bar)
            ax_bar.set_ylim(0, 1.05)
            ax_bar.set_title("Metric Score Comparison Across Models", fontsize=12, fontweight='bold')
            ax_bar.set_ylabel("Score (0.0 to 1.0)", fontweight='bold')
            ax_bar.set_xlabel("Evaluation Metric", fontweight='bold')
            ax_bar.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            ax_bar.grid(True, linestyle='--', alpha=0.6, axis='y')
            st.pyplot(fig_bar)

        with tab3:
            st.markdown("""
            ### Observations on Model Performance (Assignment Section 3 - Step 5)
            | ML Model Name | Observation about model performance |
            | :--- | :--- |
            | **Logistic Regression** | Solid linear baseline (82.07% accuracy, 0.9030 AUC). Quick to compute and interpretable, but limited in capturing non-linear interactions between customer attributes and cancellation risk. |
            | **Decision Tree** | **Overall Winner** for this dataset. Achieved top Accuracy (84.13%), highest F1 (0.7812), and highest MCC (0.6578). Effectively captures complex interaction rules (e.g. deposit type and lead time) while remaining lightweight (<25KB) and fast. |
            | **KNN** | Good accuracy (81.63%) and recall (73.37%), but inference requires distance computation to all training instances, resulting in higher latency and a large memory footprint (~7MB). |
            | **Naive Bayes** | High recall (88.25%) but poor precision (51.80%) and lowest accuracy (64.87%). The conditional independence assumption fails on correlated hotel features (lead time, ADR, party size), leading to excessive false alarms. |
            | **Support Vector Machine** | Achieved the highest AUC (0.9136) and top precision (82.88%). Non-linear RBF boundary classifies complex patterns well, but training and inference time are computationally expensive. |

            **🏆 Overall Winner for the Dataset:** **Decision Tree Classifier**
            - *Rationale:* Delivers the best balance of Accuracy, F1-Score, and MCC with minimal latency and a compact model footprint ideal for production deployment.
            """)

    # --------------------------------------------------------------------------
    # Mode B: Individual Model Detailed View
    # --------------------------------------------------------------------------
    else:
        st.subheader(f"🎯 Detailed Performance: {selected_model_name}")
        model = models[selected_model_name]

        # Compute predictions
        if not hasattr(model, 'multi_class') and hasattr(model, 'classes_'):
            setattr(model, 'multi_class', 'deprecated')
        if hasattr(model, 'predict_proba'):
            try:
                y_prob = model.predict_proba(X_test_proc)[:, 1]
            except AttributeError:
                setattr(model, 'multi_class', 'deprecated')
                y_prob = model.predict_proba(X_test_proc)[:, 1]
        elif hasattr(model, 'decision_function'):
            df_val = model.decision_function(X_test_proc)
            y_prob = 1 / (1 + np.exp(-df_val))
        else:
            y_prob = model.predict(X_test_proc)

        y_pred = (y_prob >= prob_threshold).astype(int)
        metrics = calculate_metrics_dict(y_test, y_pred, y_prob)

        # ----------------------------------------------------------------------
        # Feature C: Display of Evaluation Metrics
        # ----------------------------------------------------------------------
        st.markdown("#### 📈 Key Evaluation Metrics")
        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

        kpi1.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Accuracy</div>
            <div class="metric-value">{metrics['Accuracy']:.4f}</div>
            <div class="metric-desc">Overall Correct</div>
        </div>
        """, unsafe_allow_html=True)

        kpi2.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">ROC AUC</div>
            <div class="metric-value">{metrics['AUC']:.4f}</div>
            <div class="metric-desc">Discrimination Power</div>
        </div>
        """, unsafe_allow_html=True)

        kpi3.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Precision</div>
            <div class="metric-value">{metrics['Precision']:.4f}</div>
            <div class="metric-desc">Positive Reliability</div>
        </div>
        """, unsafe_allow_html=True)

        kpi4.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Recall</div>
            <div class="metric-value">{metrics['Recall']:.4f}</div>
            <div class="metric-desc">Catch Canceled</div>
        </div>
        """, unsafe_allow_html=True)

        kpi5.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">F1 Score</div>
            <div class="metric-value">{metrics['F1']:.4f}</div>
            <div class="metric-desc">Harmonic Mean</div>
        </div>
        """, unsafe_allow_html=True)

        kpi6.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">MCC Score</div>
            <div class="metric-value">{metrics['MCC']:.4f}</div>
            <div class="metric-desc">Matthews Corr Coeff</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # Feature D: Confusion Matrix & Classification Report
        # ----------------------------------------------------------------------
        st.markdown("#### 🔬 Diagnostic Evaluation: Confusion Matrix & Classification Report")
        col_cm, col_cr = st.columns([1.1, 1])

        with col_cm:
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            total_cases = len(y_test)

            fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.5))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                cbar=False, annot_kws={"size": 14, "weight": "bold"}
            )
            ax_cm.set_title(f"Confusion Matrix: {selected_model_name}", fontsize=12, fontweight='bold')
            ax_cm.set_xlabel('Predicted Booking Status', fontsize=10, fontweight='bold')
            ax_cm.set_ylabel('Actual Booking Status', fontsize=10, fontweight='bold')
            ax_cm.set_xticklabels(['Confirmed (0)', 'Canceled (1)'])
            ax_cm.set_yticklabels(['Confirmed (0)', 'Canceled (1)'])
            st.pyplot(fig_cm)

            st.caption(f"**Breakdown:** TN: `{tn:,}` ({tn/total_cases:.1%}) | FP: `{fp:,}` ({fp/total_cases:.1%}) | FN: `{fn:,}` ({fn/total_cases:.1%}) | TP: `{tp:,}` ({tp/total_cases:.1%})")

        with col_cr:
            st.markdown("**Detailed Classification Report:**")
            report_dict = classification_report(
                y_test, y_pred,
                target_names=['Confirmed (0)', 'Canceled (1)'],
                output_dict=True
            )
            cr_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(
                cr_df.style.format({
                    'precision': '{:.4f}',
                    'recall': '{:.4f}',
                    'f1-score': '{:.4f}',
                    'support': '{:.0f}'
                }),
                use_container_width=True
            )

            # ROC Curve for this model
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            model_auc = auc(fpr, tpr)
            fig_roc, ax_roc = plt.subplots(figsize=(5.5, 2.8))
            ax_roc.plot(fpr, tpr, color='#0d3b66', lw=2, label=f"ROC (AUC = {model_auc:.4f})")
            ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
            ax_roc.set_xlabel('False Positive Rate', fontsize=8)
            ax_roc.set_ylabel('True Positive Rate', fontsize=8)
            ax_roc.set_title(f"ROC Curve: {selected_model_name}", fontsize=9, fontweight='bold')
            ax_roc.legend(loc="lower right", fontsize=8)
            ax_roc.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig_roc)

        # ----------------------------------------------------------------------
        # Prediction Inspector
        # ----------------------------------------------------------------------
        with st.expander("🔎 Interactive Prediction Inspector (Sample Records)", expanded=False):
            sample_preview = test_df.head(15).copy()
            sample_preview['Actual Status'] = np.where(y_test.head(15) == 1, 'Canceled', 'Confirmed')
            sample_preview['Predicted Status'] = np.where(y_pred[:15] == 1, 'Canceled', 'Confirmed')
            sample_preview['Cancel Probability'] = np.round(y_prob[:15], 4)
            sample_preview['Correct?'] = np.where(y_test.head(15) == y_pred[:15], '✅ Yes', '❌ No')

            show_cols = [
                'Actual Status', 'Predicted Status', 'Cancel Probability', 'Correct?',
                'lead_time', 'deposit_type', 'market_segment', 'total_of_special_requests', 'adr'
            ]
            available_show_cols = [c for c in show_cols if c in sample_preview.columns]
            st.dataframe(sample_preview[available_show_cols], use_container_width=True)


if __name__ == '__main__':
    main()
