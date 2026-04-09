import pandas as pd
import numpy as np
import streamlit as st
import joblib
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression

class DataFrameConverter(BaseEstimator, TransformerMixin): 
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]  
        return self

    def transform(self, X, y=None):
        return pd.DataFrame(X)

sys.modules[__name__].DataFrameConverter = DataFrameConverter

model_best = joblib.load('model_best.pkl')

old_clf = model_best.named_steps['classifier']
new_clf = LogisticRegression(
    C=10,
    penalty='l1',
    solver='saga',
    max_iter=500,
    l1_ratio=0.0,
    random_state=42
)
new_clf.coef_ = old_clf.coef_
new_clf.intercept_ = old_clf.intercept_
new_clf.classes_ = old_clf.classes_
new_clf.n_iter_ = old_clf.n_iter_

# Replace the classifier in the pipeline
model_best.named_steps['classifier'] = new_clf

classifier = model_best.named_steps['classifier']
# Ensure 'multi_class' attribute exists for cross-version compatibility
# sklearn's predict_proba expects this attribute to exist in some versions
if not hasattr(classifier, 'multi_class'):
    try:
        classifier.multi_class = 'auto'
    except Exception:
        # If assignment fails for any reason, ignore to keep app running
        pass

STATUS_MAP = {
    0: "Fully Paid",
    1: "In Grace Period",
    2: "Late (16-30 days)",
    3: "Late (31-120 days)",
    4: "Default",
    5: "Charged Off"
}

RISK_LEVEL = {
    0: "low",
    1: "medium",
    2: "medium",
    3: "high",
    4: "high",
    5: "high"
}

RISK_COLOR = {
    "low": "green",
    "medium": "orange",
    "high": "red"
}

def run():
    st.set_page_config(page_title="Loan Defend", page_icon="💳", layout="centered")
    st.title("💳 Loan Defend — Loan Status Predictor")
    st.caption("Predict loan repayment status using machine learning.")

    with st.form(key='loan_form'):

        st.subheader("Borrower Profile")
        name = st.text_input("Full Name of Borrower")

        col1, col2 = st.columns(2)
        with col1:
            grade = st.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"])
            emp_status = st.selectbox("Employment Status", ["employed", "unemployed"])
        with col2:
            verification_status = st.selectbox("Verification Status", ["Not Verified", "Source Verified", "Verified"])
            purpose = st.selectbox("Purpose", [
                "debt_consolidation", "credit_card", "home_improvement",
                "other", "major_purchase", "small_business", "medical",
                "car", "moving", "house", "vacation", "wedding", "renewable_energy"
            ])

        st.divider()
        st.subheader("Loan Details")

        col1, col2, col3 = st.columns(3)
        with col1:
            loan_amnt = st.number_input("Loan Amount ($)", min_value=0.0, step=100.0)
            dti = st.number_input("DTI", min_value=0.0, step=0.1)
        with col2:
            term = st.number_input("Term (months)", min_value=0, step=1)
            delinq_2yrs = st.number_input("Delinquencies (2yr)", min_value=0, step=1)
        with col3:
            int_rate = st.number_input("Interest Rate (%)", min_value=0.0, step=0.1)
            mths_since_last_delinq = st.number_input("Mths Since Last Delinq", min_value=0, step=1)

        st.divider()
        st.subheader("Outstanding & Payments")

        col1, col2, col3 = st.columns(3)
        with col1:
            out_prncp = st.number_input("Outstanding Principal ($)", min_value=0.0, step=100.0)
            total_pymnt = st.number_input("Total Payment ($)", min_value=0.0, step=100.0)
            total_rec_prncp = st.number_input("Total Rec. Principal ($)", min_value=0.0, step=100.0)
        with col2:
            out_prncp_inv = st.number_input("Out. Principal Inv. ($)", min_value=0.0, step=100.0)
            total_pymnt_inv = st.number_input("Total Payment Inv. ($)", min_value=0.0, step=100.0)
            total_rec_int = st.number_input("Total Rec. Interest ($)", min_value=0.0, step=100.0)

        st.divider()
        st.subheader("Recovery & History")

        col1, col2 = st.columns(2)
        with col1:
            recoveries = st.number_input("Recoveries ($)", min_value=0.0, step=10.0)
            last_pymnt_amnt = st.number_input("Last Payment Amount ($)", min_value=0.0, step=10.0)
        with col2:
            collection_recovery_fee = st.number_input("Collection Recovery Fee ($)", min_value=0.0, step=10.0)
            mths_since_last_major_derog = st.number_input("Mths Since Last Major Derog", min_value=0, step=1)

        submit = st.form_submit_button("Run Prediction", use_container_width=True)

    if submit:
        if not name.strip():
            st.error("Please enter the borrower's name.")
            return

        # Build input in exact column order: significant_cat + significant_num
        input_data = pd.DataFrame([{
            'grade': grade,
            'verification_status': verification_status,
            'purpose': purpose,
            'emp_status': emp_status,
            'loan_amnt': loan_amnt,
            'term': term,
            'int_rate': int_rate,
            'dti': dti,
            'delinq_2yrs': delinq_2yrs,
            'mths_since_last_delinq': mths_since_last_delinq,
            'out_prncp': out_prncp,
            'out_prncp_inv': out_prncp_inv,
            'total_pymnt': total_pymnt,
            'total_pymnt_inv': total_pymnt_inv,
            'total_rec_prncp': total_rec_prncp,
            'total_rec_int': total_rec_int,
            'recoveries': recoveries,
            'collection_recovery_fee': collection_recovery_fee,
            'last_pymnt_amnt': last_pymnt_amnt,
            'mths_since_last_major_derog': mths_since_last_major_derog,
        }])

        prediction = int(model_best.predict(input_data)[0])
        proba = model_best.predict_proba(input_data)[0]
        status = STATUS_MAP[prediction]
        risk = RISK_LEVEL[prediction]
        confidence = round(float(np.max(proba)) * 100, 2)

        st.divider()
        st.subheader("Prediction Result")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**{name}** is predicted to be:")
            st.markdown(f"### {status}")
            st.markdown(f":{RISK_COLOR[risk]}[{risk.capitalize()} Risk]")
        with col2:
            st.metric("Confidence", f"{confidence}%")

        st.divider()
        st.markdown("**Class Probabilities**")
        prob_df = pd.DataFrame({
            'Status': list(STATUS_MAP.values()),
            'Probability (%)': [round(float(p) * 100, 2) for p in proba]
        }).set_index('Status')
        st.bar_chart(prob_df)

if __name__ == '__main__':
    run()