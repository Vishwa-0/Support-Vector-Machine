import streamlit as st
import numpy as np
import joblib

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Iris Species Intelligence",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------- Styling ----------------
st.markdown("""
<style>
body {
    background-color: #020617;
}
.card {
    background: rgba(30,41,59,0.75);
    backdrop-filter: blur(10px);
    padding: 2rem;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 1.5rem;
}
.result {
    background: linear-gradient(135deg,#16a34a,#15803d);
    padding: 1.6rem;
    border-radius: 16px;
    text-align: center;
}
h1, h2, h3 {
    color: #e5e7eb;
}
p {
    color: #cbd5e1;
}
.metric {
    font-size: 1.4rem;
    font-weight: 600;
}
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return joblib.load("iris_svm_model.joblib")

model = load_model()

# ---------------- Header ----------------
st.markdown("""
<div class="card">
    <h1>Iris Species Intelligence</h1>
    <p>
        Classify iris flowers using a <b>Linear Support Vector Machine</b>.
        Designed for educational diagnostics and model demonstration.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- Input Section ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.header("Morphological Measurements (cm)")

sepal_length = st.slider(
    "Sepal Length",
    min_value=4.0,
    max_value=8.0,
    value=5.8,
    step=0.1
)

sepal_width = st.slider(
    "Sepal Width",
    min_value=2.0,
    max_value=4.5,
    value=3.0,
    step=0.1
)

petal_length = st.slider(
    "Petal Length",
    min_value=1.0,
    max_value=7.0,
    value=4.0,
    step=0.1
)

petal_width = st.slider(
    "Petal Width",
    min_value=0.1,
    max_value=2.5,
    value=1.3,
    step=0.1
)

predict = st.button("Classify Species", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Prediction ----------------
if predict:
    user_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(user_data)[0]

    species_descriptions = {
        "setosa": "Small petals, compact structure, high separability",
        "versicolor": "Moderate petal size, transitional morphology",
        "virginica": "Large petals, elongated structure"
    }

    description = species_descriptions.get(prediction.lower(), "Unknown profile")

    st.markdown(
        f"""
        <div class="result">
            <h2>Predicted Species</h2>
            <p class="metric">{prediction}</p>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- Footer ----------------
st.markdown("""
<div class="footer">
    Linear SVM • Multiclass Classification • Iris Dataset
</div>
""", unsafe_allow_html=True)
