import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('banknote_auth_model.joblib')

st.title('Bank Note Authenticator')

st.write("""
This app uses machine learning to predict whether a bank note is authentic or counterfeit based on its features.
""")

# Create input fields for the four features
variance = st.number_input('Variance of Wavelet Transformed Image', format="%.6f")
skewness = st.number_input('Skewness of Wavelet Transformed Image', format="%.6f")
curtosis = st.number_input('Curtosis of Wavelet Transformed Image', format="%.6f")
entropy = st.number_input('Entropy of Image', format="%.6f")

# When the 'Predict' button is clicked
if st.button('Predict'):
    # Make a prediction
    features = np.array([[variance, skewness, curtosis, entropy]])
    prediction = model.predict(features)
    
    # Display the prediction
    if prediction[0] == 0:
        st.error('The bank note is predicted to be counterfeit.')
    else:
        st.success('The bank note is predicted to be authentic.')
    
    # Display prediction probability
    proba = model.predict_proba(features)
    st.write(f'Probability of being authentic: {proba[0][1]:.2%}')
    st.write(f'Probability of being counterfeit: {proba[0][0]:.2%}')

st.write("""
### Feature Information:
- Variance: Variance of Wavelet Transformed image (continuous)
- Skewness: Skewness of Wavelet Transformed image (continuous)
- Curtosis: Curtosis of Wavelet Transformed image (continuous)
- Entropy: Entropy of image (continuous)
""")