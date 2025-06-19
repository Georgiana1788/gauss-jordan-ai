
import streamlit as st
from retea_gauss_jordan_complet_CORECTAT import faza1_train_model, faza1_demo, faza2_train_model, faza2_demo, faza3_train_model, faza3_demo

st.set_page_config(page_title="Gauss-Jordan AI", layout="centered")

# Select limba
lang = st.selectbox("🌐 Limba / Language", ["Română", "English"])

# Texte în funcție de limbă
TEXT = {
    "Română": {
        "title": "🧠 Gauss-Jordan AI – Soluționare ecuații liniare",
        "select_phase": "Alege faza:",
        "phases": ["Faza 1 – ax = b", "Faza 2 – ax + by = c", "Faza 3 – Sistem 3x3"],
        "btn": "🔍 Rezolvă",
        "success": "Modelul este antrenat. Afișăm demonstrația mai jos:",
        "a": "Coeficient a",
        "b": "Coeficient b",
        "x_real": "Valoare reală x",
        "y_real": "Valoare reală y",
        "z_real": "Valoare reală z"
    },
    "English": {
        "title": "🧠 Gauss-Jordan AI – Solving Linear Equations",
        "select_phase": "Select phase:",
        "phases": ["Phase 1 – ax = b", "Phase 2 – ax + by = c", "Phase 3 – 3x3 system"],
        "btn": "🔍 Solve",
        "success": "Model trained. Showing the explanation below:",
        "a": "Coefficient a",
        "b": "Coefficient b",
        "x_real": "True value of x",
        "y_real": "True value of y",
        "z_real": "True value of z"
    }
}[lang]

st.title(TEXT["title"])

phase = st.selectbox(TEXT["select_phase"], TEXT["phases"])

if phase == TEXT["phases"][0]:  # Faza 1
    a = st.number_input(TEXT["a"], value=3.0)
    x_real = st.number_input(TEXT["x_real"], value=5.0)
    if st.button(TEXT["btn"]):
        model = faza1_train_model()
        st.success(TEXT["success"])
        faza1_demo(model, a, x_real)

elif phase == TEXT["phases"][1]:  # Faza 2
    a = st.number_input(TEXT["a"], value=2.0)
    b = st.number_input(TEXT["b"], value=3.0)
    x_real = st.number_input(TEXT["x_real"], value=4.0)
    y_real = st.number_input(TEXT["y_real"], value=5.0)
    if st.button(TEXT["btn"]):
        model = faza2_train_model()
        st.success(TEXT["success"])
        faza2_demo(model, a, b, x_real, y_real)

elif phase == TEXT["phases"][2]:  # Faza 3
    x_real = st.number_input(TEXT["x_real"], value=2.0)
    y_real = st.number_input(TEXT["y_real"], value=3.0)
    z_real = st.number_input(TEXT["z_real"], value=4.0)
    if st.button(TEXT["btn"]):
        model = faza3_train_model()
        st.success(TEXT["success"])
        faza3_demo(model, x_real, y_real, z_real)
