import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
from utils import predict_customer_churn,predict_customer_churn_array

model = pickle.load(open("XGBoost-Model.pkl", "rb"))
top_feats = pickle.load(open("Top-Features.pkl", "rb"))
mean_values = pickle.load(open("Mean-Vals.pkl", "rb"))
higher_features = pickle.load(open("High-Features.pkl", "rb"))
lower_features = pickle.load(open("Low-Features.pkl", "rb"))

st.title("Customer Churn Prediction with XAI Insights")

option = st.radio("Choose Prediction Type", ("Single Prediction", "Batch Prediction"),index=None)

if option == "Single Prediction":

    st.subheader('Enter Customer Details ', divider='red')

    col1, col2 = st.columns(2)

    with col1:
        equip_days = st.number_input("Days with Current Equipment",
                                     value=None,
                                     placeholder="Enter current equipment days",
                                     help="Number of days the current equipment has been in use")

        months = st.number_input("Months in Service",
                                 value=None,
                                 placeholder="Enter the amount of months in service so far",
                                 help="Total number of months the service has been active")

        off_peak_calls = st.number_input("Off-Peak Calls In/Out",
                                         value=None,
                                         placeholder="Enter off-peak calls made or received",
                                         help="Number of off-peak calls made or received")

        total_charge = st.number_input("Total Recurring Charge",
                                       value=None,
                                       placeholder="Enter total recurring charges",
                                       help="Total recurring charges for the service")

    with col2:
        unanswered_calls = st.number_input("Unanswered Calls",
                                           value=None,
                                           placeholder="Enter number of unanswered calls",
                                           help="Number of unanswered calls")

        income_group = st.number_input("Income Group",
                                       value=None,
                                       placeholder="Enter income group",
                                       help="Income group of the customer")

        opt_out_mailings = st.selectbox("Opt-Out Mailings",
                                        ('Yes','No'),
                                        index=None,
                                        help="Was there an opt-out email?")

        monthly_minutes = st.number_input("Monthly Minutes",
                                          value=None,
                                          placeholder="Enter total minutes used in a month",
                                          help="Total minutes used in a month")

    # Define the layout for the remaining input fields in two columns per row
    col3, col4 = st.columns(2)

    with col3:
        perc_change_minutes = st.number_input("Percentage Change in Minutes",
                                              value=None,
                                              placeholder="Enter percentage change in minutes",
                                              help="Percentage change in minutes compared to previous usage")

    with col4:
        retention_calls = st.number_input("Retention Calls",
                                          value=None,
                                          placeholder="Enter number of retention calls",
                                          help="Number of calls related to customer retention")

    # Add a button to trigger the prediction
    sample = [equip_days, months, off_peak_calls, total_charge, unanswered_calls,
                        income_group, opt_out_mailings, monthly_minutes, perc_change_minutes,
                        retention_calls]

    st.subheader('Prediction', divider='violet')

    if st.button('Predict Churn'):

        if None in sample:
            st.error('Please provide all inputs')
        else:
            predict_customer_churn(sample)
else:
    st.subheader('Enter Batch Customer Details', divider='red')
    st.info("""
        **Batch Prediction Input Guidelines:**
        - Do not include Customer ID or Churn.
        - Maintain the order of columns as in the original dataframe.
        - Avoid using NULL values; if necessary, replace NULL with 0.
    """)
    batch_input = st.text_area("Enter the customer details as a list of arrays (e.g., [[X1,X2,X3],"
                               "                             [X1,X2,X3],"
                               "                             [X1,X2,X3]])",height=300)

    if not batch_input:
        st.warning('Please enter the customers details as a list of arrays in the correct format')
        if st.button('Predict Churn for Batch'):
            st.error('Refer to the warning!')

    else:
        try:
            batch_data = eval(batch_input)
            if st.button('Predict Churn for Batch'):
                predict_customer_churn_array(batch_data)
        except Exception as e:
            st.error(f'Error: {e}. Please enter the list of arrays in the correct format')