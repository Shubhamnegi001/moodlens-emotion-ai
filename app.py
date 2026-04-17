import streamlit as st
import pickle
import os

# ---------- Load Files ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "tfidf.pkl"), "rb"))
label_map = pickle.load(open(os.path.join(BASE_DIR, "label_map.pkl"), "rb"))

# ---------- Page Config ----------
st.set_page_config(page_title="MoodLens AI", page_icon="🧠", layout="centered")

# ---------- Styling ----------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.title("🧠 MoodLens AI")
st.markdown("### NLP-Based Emotion Detection System")

# ---------- Sidebar ----------
st.sidebar.title("About")
st.sidebar.info("""
MoodLens AI analyzes text and predicts emotional tone using:
- TF-IDF Vectorization
- Stacking Classifier  
Accuracy: ~90.6%
""")

# ---------- Input ----------
user_input = st.text_area("💬 Enter your text:")

# ---------- Prediction ----------
if st.button("🔍 Analyze Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        vec = vectorizer.transform([user_input])
        pred = model.predict(vec)[0]
        emotion = label_map[pred]

        # ---------- Result ----------
        st.success(f"🎯 Detected Emotion: **{emotion.upper()}**")

        # ---------- Emoji ----------
        emoji_dict = {
            "joy": "😄",
            "anger": "😡",
            "sadness": "😢",
            "fear": "😨",
            "love": "❤️",
            "surprise": "😲"
        }

        if emotion in emoji_dict:
            st.markdown(f"### {emoji_dict[emotion]}")

        # ---------- Professional Note ----------
        st.markdown("""
        <small style='color: gray;'>
        ⚠️ Note: Certain emotions such as <b>joy</b> and <b>love</b> may exhibit semantic overlap. 
        The model predicts the most probable class based on learned patterns from training data.
        </small>
        """, unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("---")
st.markdown("<center><small>Built with ❤️ using NLP, Machine Learning & Streamlit</small></center>", unsafe_allow_html=True)