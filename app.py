
import streamlit as st
import joblib
import os

# Configure page
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# Load trained model
MODEL_PATH = "model/fake_news_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Please train the model first.")
    st.stop()

model = load_model()

# Application heading
st.title("📰 Fake News Detection System")

st.markdown(
    "### Detect potentially fake news using Machine Learning and NLP."
)

st.write(
    "Enter a news article below to classify it using our trained model."
)

# Input section
news_text = st.text_area(
    "Enter News Article",
    placeholder="Paste your news article here...",
    height=250
)

# Prediction button
if st.button("Analyze News", type="primary"):

    if news_text.strip():

        prediction = model.predict([news_text])[0]

        probabilities = model.predict_proba([news_text])[0]

        classes = model.classes_
        confidence = max(probabilities) * 100

        st.subheader("Prediction Result")

        # IMPORTANT:
        # Verify label mapping from your dataset.
        if prediction == 1:
            st.success("Prediction: REAL NEWS")
        else:
            st.error("Prediction: FAKE NEWS")

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(float(confidence / 100))

        st.caption(
            "Confidence is the model's estimated probability, "
            "not proof that the news is authentic."
        )

    else:
        st.warning("Please enter a news article.")

# Footer
st.divider()

st.caption(
    "Fake News Detection System | Python | NLP | Machine Learning"
)

