import streamlit as st
import joblib
import re
import string

# ==========================
# CONFIGURATION DE LA PAGE
# ==========================
st.set_page_config(
    page_title="IT Ticket Classifier",
    page_icon="🎯",
    layout="centered",
)

# ==========================
# CHARGEMENT DES RESSOURCES
# ==========================
@st.cache_resource
def load_resources():
    model = joblib.load("ticket_classifier_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf

model, tfidf = load_resources()

# ==========================
# FONCTION DE NETTOYAGE
# ==========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ==========================
# BARRE LATÉRALE
# ==========================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3145/3145825.png", width=120)
    st.markdown("## 🧠 À propos du projet")
    st.write("""
    Cette application classe automatiquement les tickets informatiques en catégories grâce à :
    - **TF-IDF** pour la vectorisation du texte  
    - **Régression Logistique** pour la prédiction  
    - **Streamlit** pour l’interface utilisateur
    """)
    st.markdown("---")
    st.markdown("👨‍💻 **Projet réalisé par Elvis Obounou Zolo**")
    st.caption("Étudiant en IA & Data — Aivancity Paris-Cachan")

# ==========================
# CONTENU PRINCIPAL
# ==========================
st.image("https://cdn-icons-png.flaticon.com/512/3221/3221897.png", width=100)
st.title("Classificateur de tickets d’assistance informatique")

st.write(
    "Cette application utilise l’apprentissage automatique (TF-IDF + Régression Logistique) "
    "pour classer automatiquement les tickets de support IT selon leur contenu."
)

# Champ de texte
user_input = st.text_area(
    "✏️ Entrez la description de votre ticket ci-dessous :",
    placeholder="par exemple : impossible de se connecter au VPN après la mise à jour de Windows...",
)

# Bouton de prédiction
if st.button("🔍 Prédire la catégorie"):
    if user_input.strip() == "":
        st.warning("Veuillez entrer une description avant de lancer la prédiction.")
    else:
        cleaned_text = clean_text(user_input)
        vectorized = tfidf.transform([cleaned_text])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized).max() * 100

        st.success(f"✅ Catégorie prédite : **{prediction}**")
        st.write(f"**Confiance du modèle :** {confidence:.2f}%")

# Pied de page
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>🧩 Développé par <b>Elvis Obounou Zolo</b> — "
    "Analyste de données & Étudiant en IA (Aivancity)</p>",
    unsafe_allow_html=True,
)
