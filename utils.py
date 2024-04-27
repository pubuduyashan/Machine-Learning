import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap


loaded_model = pickle.load(open("XGBoost-Model.pkl", "rb"))
top_feats = pickle.load(open("Top-Features.pkl", "rb"))
mean_values = pickle.load(open("Mean-Vals.pkl", "rb"))
higher_features = pickle.load(open("High-Features.pkl", "rb"))
lower_features = pickle.load(open("Low-Features.pkl", "rb"))
churn_insights = pickle.load(open("Churn-Insights.pkl", "rb"))
non_churn_insights = pickle.load(open("NonChurn-Insights.pkl", "rb"))
preprocessor = pickle.load(open('Transformer.pkl', 'rb'))
feats = pickle.load(open("Feats.pkl", "rb"))
prefeats = pickle.load(open("PreFeats.pkl", "rb"))
postfeats = pickle.load(open("PostFeats.pkl", "rb"))


def predict_customer_churn(sample):
    # Reshape the input sample
    sample = np.array(sample).reshape(1, -1)
    sample = pd.DataFrame(sample,columns=top_feats)

    sample['OptOutMailings'] = sample['OptOutMailings'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Convert DataFrame to float type
    sample = sample.astype(float)

    # Predict using the loaded model
    prediction = loaded_model.predict(sample)[0]

    # Explain the prediction using SHAP values
    explainer = shap.TreeExplainer(loaded_model)
    sample_shap_values = explainer.shap_values(sample)

    # Create a dictionary to store SHAP values for features
    shap_dict = {}

    # Zip feature names with their SHAP values and sort them
    zipped_shape = list(zip(top_feats, sample_shap_values[0]))
    sorted_zipped_shape = sorted(zipped_shape, key=lambda x: abs(x[1]), reverse=True)

    # Populate the shap_dict with feature names and SHAP values
    for feature, shap_val in sorted_zipped_shape:
        shap_dict[feature] = shap_val

    # Display prediction result using st.warning or st.success
    if prediction == 1:
        st.warning('**Prediction:** The customer is likely to churn')
    elif prediction == 0:
        st.success('**Prediction:** The customer is unlikely to churn')

    st.subheader('XAI Insights on Prediction', divider='blue')

    # Display SHAP values and explanations
    for feature, sval in shap_dict.items():
        if prediction == 1:
            if (feature in higher_features) and (sample[feature][0] > mean_values[feature]) and (sval > 0):
                st.markdown(f"* {churn_insights['higher_features'][feature]}")
            if (feature in lower_features) and (sample[feature][0] < mean_values[feature]) and (sval > 0):
                st.markdown(f"* {churn_insights['lower_features'][feature]}")
        elif prediction == 0:
            if (feature in higher_features) and (sample[feature][0] < mean_values[feature]) and (sval < 0):
                st.markdown(f"* {non_churn_insights['higher_features'][feature]}")
            if (feature in lower_features) and (sample[feature][0] > mean_values[feature]) and (sval < 0):
                st.markdown(f"* {non_churn_insights['lower_features'][feature]}")

    return prediction, sample

def predict_customer_churn_array(samples):
    predictions = []
    samps = []
    for i,sample in enumerate(samples):
        sample = np.array(sample).reshape(1, -1)


        sample = pd.DataFrame(sample, columns=prefeats)


        sample = sample[feats]

        sample = preprocessor.transform(sample)
        sample = pd.DataFrame(sample, columns=postfeats)
        st.write(sample)

        # Create a DataFrame with the sample data and column names
        sample = sample[top_feats]

        # Convert 'OptOutMailings' column to binary
        sample['OptOutMailings'] = sample['OptOutMailings'].apply(lambda x: 1 if x == 'Yes' else 0)

        # Convert DataFrame to float type
        sample = sample.astype(float)
        samps.append(sample)

        # Predict using the loaded model
        prediction = loaded_model.predict(sample)[0]
        predictions.append(prediction)

        # Explain the prediction using SHAP values
        explainer = shap.TreeExplainer(loaded_model)
        sample_shap_values = explainer.shap_values(sample)

        # Create a dictionary to store SHAP values for features
        shap_dict = {}

        # Zip feature names with their SHAP values and sort them
        zipped_shape = list(zip(top_feats, sample_shap_values[0]))
        sorted_zipped_shape = sorted(zipped_shape, key=lambda x: abs(x[1]), reverse=True)

        # Populate the shap_dict with feature names and SHAP values
        for feature, shap_val in sorted_zipped_shape:
            shap_dict[feature] = shap_val

        # Display prediction result using st.warning or st.success
        if prediction == 1:
            st.warning(f'**Prediction:** Customer {i} is likely to churn')
        elif prediction == 0:
            st.success(f'**Prediction:** Customer {i} is unlikely to churn')

        st.subheader('XAI Insights on Prediction', divider='blue')

        # Display SHAP values and explanations
        for feature, sval in shap_dict.items():
            if prediction == 1:
                if (feature in higher_features) and (sample[feature][0] > mean_values[feature]) and (sval > 0):
                    st.markdown(f"* {churn_insights['higher_features'][feature]}")
                if (feature in lower_features) and (sample[feature][0] < mean_values[feature]) and (sval > 0):
                    st.markdown(f"* {churn_insights['lower_features'][feature]}")
            elif prediction == 0:
                if (feature in higher_features) and (sample[feature][0] < mean_values[feature]) and (sval < 0):
                    st.markdown(f"* {non_churn_insights['higher_features'][feature]}")
                if (feature in lower_features) and (sample[feature][0] > mean_values[feature]) and (sval < 0):
                    st.markdown(f"* {non_churn_insights['lower_features'][feature]}")

    return predictions, samps
