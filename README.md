# Hotel Booking Demand Prediction & Classification System

**Work Integrated Learning Programmes (WILP) Division**  
**M.Tech (AIML) — Machine Learning Assignment - 2**  
**Course Code:** Machine Learning  

---

## a. Problem Statement

Booking cancellations represent a substantial operational inefficiency and financial vulnerability for the hospitality industry. When a guest cancels a reservation late, hotel revenue management systems are forced to either leave rooms vacant or discount prices heavily on short notice, eroding RevPAR (Revenue Per Available Room) and complicating inventory and staffing forecasts.

The objective of this project is to develop, evaluate, and deploy an end-to-end Machine Learning classification solution capable of predicting whether an incoming hotel reservation is likely to be canceled (`is_canceled` = 1) or confirmed (`is_canceled` = 0) prior to the arrival date. By accurately identifying high-risk cancellation reservations at booking time, hotel operators can:
1. Implement targeted proactive retention policies (e.g., personalized customer service, confirmation calls, incentive packages).
2. Optimize dynamic pricing and automated overbooking strategies with minimal walk-ratio risk.
3. Eliminate target leakage features to ensure real-world operational validity.

---

## b. Dataset Description [1 Mark]

The dataset utilized is the benchmark **Hotel Booking Demand Dataset** (sourced from public repository Kaggle / Antonio, de Almeida and Nunes, 2019). It captures actual bookings for two hotel properties located in Portugal: a Resort Hotel (H1, located in the Algarve resort region) and a City Hotel (H2, located in Lisbon).

- **Total Original Instances:** 119,390 reservations
- **Target Variable:** `is_canceled` (Binary classification: `0` = Confirmed/Check-Out, `1` = Canceled/No-Show)
- **Class Distribution:** 62.96% Confirmed (0) vs. 37.04% Canceled (1)
- **Total Features Analyzed:** 31 explanatory features across guest demographics, booking timeline, room configurations, distribution channels, and seasonal attributes.

### Key Preprocessing & Domain Feature Engineering:
1. **Missing Value Imputation:**
   - `country`: Imputed with `'Unknown'` (represented guest origins without registered country codes).
   - `agent` and `company`: Imputed with `0` (indicating private direct bookings not routed via corporate or external agencies).
   - `children`: Missing values imputed with `0`.
2. **Filtering Out Infeasible Outliers:**
   - Excluded reservations where total guest count (`adults + children + babies`) equals `0`.
   - Filtered out erroneous negative average daily rate (ADR) and extreme pricing anomalies (`adr >= 5000`).
3. **Engineered Domain Features:**
   - `total_stay_nights`: Sum of `stays_in_weekend_nights` and `stays_in_week_nights`.
   - `total_guests`: Aggregate guest party size (`adults + children + babies`).
   - `room_type_changed`: Binary flag indicating if the room assigned differs from the room reserved (captures room upgrades/reallocations).
   - `is_family`: Binary indicator for travelers with children or infants.
   - `country_group`: Top 10 inbound traveler countries (PRT, GBR, FRA, ESP, DEU, ITA, IRL, BEL, BRA, NLD) encoded explicitly with other origins aggregated into `'Other'`.
4. **Data Leakage Mitigation:**
   - Strictly dropped `reservation_status` ('Check-Out', 'Canceled', 'No-Show') and `reservation_status_date` as they are created *after* the outcome occurs and represent direct target leakage.
   - Excluded `assigned_room_type` (retained `room_type_changed` indicator) and `arrival_date_year` to promote temporal generalization.
5. **Standardization & Encoding:**
   - Continuous numerical features standardized via `StandardScaler`.
   - Categorical attributes encoded via `OneHotEncoder(handle_unknown='ignore')`.

---

## c. GitHub Repository Link [1 Mark]

- **GitHub Repository:** [https://github.com/kameswarasastry59-max/Hotel_Booking_Demand](https://github.com/kameswarasastry59-max/Hotel_Booking_Demand)
- **Live Streamlit Web App:** *(Deployed on Streamlit Community Cloud)*

### Repository Directory Structure
```text
Hotel_Booking_Demand/
│-- app.py                              # Interactive Streamlit application
│-- requirements.txt                    # Project dependencies
│-- README.md                           # Comprehensive documentation & evaluation report
│-- test_data.csv                       # Formatted test dataset (3,000 instances) for web app evaluation
│-- model/                              # Saved model artifacts & pipeline
│   │-- preprocessor.joblib             # Fitted ColumnTransformer (Scaler + OneHotEncoder)
│   │-- feature_columns.json            # Feature schema metadata
│   │-- logistic_regression.joblib      # Trained Logistic Regression model
│   │-- decision_tree.joblib            # Trained Decision Tree Classifier
│   │-- knn.joblib                      # Trained K-Nearest Neighbor Classifier
│   │-- naive_bayes.joblib              # Trained Gaussian Naive Bayes Classifier
│   │-- svm.joblib                      # Trained Support Vector Machine (RBF)
│   │-- comparison_table.csv            # Tabulated metric comparisons
│   │-- metrics_summary.json            # Machine-readable evaluation metrics
│   │-- train_models.py                 # Reproducible training and evaluation pipeline
│   └── Model_Training_and_Evaluation.ipynb
│-- notebooks/
│   │-- Exploratory_Data_Analysis.ipynb # 13-step comprehensive EDA notebook
│   └── Model_Training_and_Evaluation.ipynb # Complete 5-model training & diagnostics
└── data/
    └── hotel_bookings.csv              # Raw hotel bookings dataset
```

---

## d. Models Used & Evaluation Metrics [5 Marks]

All five classification models were trained and tested on the exact same benchmark split (80% stratified train, 20% test = 3,000 holdout instances). The 6 mandatory evaluation metrics are recorded below:

### Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8207 | 0.9030 | 0.8056 | 0.6866 | 0.7413 | 0.6100 |
| **Decision Tree** | **0.8413** | 0.9127 | 0.8072 | 0.7569 | **0.7812** | **0.6578** |
| **KNN** | 0.8163 | 0.8879 | 0.7658 | 0.7337 | 0.7494 | 0.6049 |
| **Naive Bayes** | 0.6487 | 0.8269 | 0.5180 | **0.8825** | 0.6528 | 0.3939 |
| **Support Vector Machine** | 0.8370 | **0.9136** | **0.8288** | 0.7115 | 0.7657 | 0.6462 |

---

## Observations on Model Performance [3 Marks]

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Acts as an effective, highly interpretable linear baseline. It achieves strong accuracy (82.07%) and area under ROC (0.9030). Training is near instantaneous (<0.4s). However, because it fits a single hyper-plane, it under-captures complex non-linear feature interactions (such as the non-linear interaction between deposit type, booking changes, and lead time), resulting in a lower recall of 68.66%. |
| **Decision Tree** | Delivers the best overall balanced performance across the dataset. Pre-pruning (`max_depth=10`, `min_samples_split=20`, `min_samples_leaf=10`) effectively curbed variance while preserving critical hierarchical conditional rules. It achieves the highest Accuracy (84.13%), highest F1-Score (0.7812), and highest Matthews Correlation Coefficient (0.6578), while operating with an ultra-lightweight footprint (<25 KB). |
| **KNN** | Achieves solid baseline performance (81.63% accuracy and 0.8879 AUC) by clustering nearest behavioral neighbors. However, inference latency scales with sample size ($O(N \cdot D)$) and the model requires storing the entire training set (7.05 MB), making it less memory-efficient and slower for real-time web deployment. |
| **Naive Bayes** | Yields the highest Recall (88.25%), effectively detecting almost nine out of ten cancellations. However, this comes at the expense of high false positives (Precision: 51.80%, Accuracy: 64.87%). The conditional feature independence assumption is severely violated by correlated hotel attributes (e.g., stay nights, party size, ADR), causing it to over-predict cancellations. |
| **Support Vector Machine** | Employs an RBF kernel to capture non-linear manifolds, achieving the highest discriminatory power (AUC: 0.9136) and the highest Precision (82.88%). When the SVM predicts a cancellation, it is highly reliable. However, the quadratic optimization complexity makes training and Platt probability scaling significantly more compute-intensive (140.3s). |

### Overall Winner for Your Dataset?
**🏆 Overall Winner: Decision Tree Classifier**

#### Analytical Justification:
1. **Superior Balanced Metrics:** Decision Tree obtained the highest **Accuracy (84.13%)**, highest **F1 Score (0.7812)**, and highest **Matthews Correlation Coefficient (MCC: 0.6578)** across all tested models, with an outstanding **AUC of 0.9127**.
2. **Minimal Computational Footprint:** Unlike KNN and SVM, Decision Tree executes predictions in microseconds with virtually zero memory overhead (<25 KB artifact size). This makes it the most robust candidate for deployment on resource-constrained cloud platforms like Streamlit Community Cloud.
3. **Interpretability & Business Value:** Decision tree decision paths provide direct, auditable rules (e.g., threshold conditions on deposit type, lead time, and customer segment) that hotel revenue managers can translate into clear operational policies.

---

## Interactive Streamlit Web Application

The interactive web application (`app.py`) satisfies all four mandatory interface features:
1. **Dataset Upload Option (CSV):** Allows users to upload their own test dataset CSV (containing input features and `is_canceled`). A sample dataset download button and automatic fallback to `test_data.csv` ensure zero downtime and instant evaluation.
2. **Model Selection Dropdown:** Seamlessly switch between individual models or select *"All Models (Comparative Benchmark)"* for side-by-side assessment.
3. **Display of Evaluation Metrics:** Renders clear KPI metric cards for Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
4. **Confusion Matrix & Classification Report:** Interactive heatmap visualization with labeled True/False Positive/Negative counts and complete class-level precision/recall breakdown, plus ROC curves and prediction inspection.

### Running Locally
```bash
# 1. Clone the repository
git clone https://github.com/kameswarasastry59-max/Hotel_Booking_Demand.git
cd Hotel_Booking_Demand

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit application
streamlit run app.py
```

---

## Deployment on Streamlit Cloud

The interactive web application is deployed and publicly accessible on Streamlit Cloud at the following live link:
https://kameswarasastry59-max-hotel-booking-demand-app-dvejoz.streamlit.app/
