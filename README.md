# 🎯 Ticket Classifier App

Projet de classification automatique de tickets d’assistance IT à l’aide du Machine Learning.  
L’application, développée avec **Streamlit**, prédit la catégorie d’un ticket saisi par l’utilisateur (ex : matériel, accès, support RH…) et affiche le **niveau de confiance du modèle**.

---

## 🧠 Objectif
Automatiser le tri des tickets IT pour aider les équipes support à gagner du temps et à prioriser leurs actions.

---

## ⚙️ Tech Stack
- **Langage :** Python  
- **Modèle :** Régression Logistique (TF-IDF, scikit-learn)  
- **Interface :** Streamlit  
- **Librairies :** pandas, numpy, joblib, re, string  
- **Déploiement :** Streamlit Cloud  

---

## 📊 Résultats
- **Accuracy :** 85.3 %  
- Très bons scores sur les catégories *Purchase*, *Hardware* et *Access*  
- Dataset : *IT Support Ticket Classification (Kaggle)*, 47 837 tickets  

---

## 🖥️ Aperçu
> **Exemple :** “Unable to connect to printer after update”  
> **→ Catégorie prédite :** Hardware (Confiance : 85.6 %)

---

## 👤 Auteur

**Elvis Obounou Zolo**  
Étudiant en Master 1 à [Aivancity Paris-Cachan](https://www.aivancity.ai/) – Grande École de l’IA et de la Data reconnue par l’État.  
Passionné par la Data Science, le Machine Learning et les projets d’automatisation de la donnée.

📧 [bitamvillage@gmail.com](mailto:bitamvillage@gmail.com)  
💼 [linkedin.com/in/elvis-obounou](https://linkedin.com/in/elvis-obounou)  

---

## 🌐 Démo
👉 [Lancer l’application Streamlit](https://ticket-classifier-app-iirq2upuzukxaqdcrjefdw.streamlit.app/)
