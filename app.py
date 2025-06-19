
import streamlit as st
from retea_gauss_jordan_complet_CORECTAT import faza1_train_model, faza1_demo

st.set_page_config(page_title="Gauss-Jordan AI", layout="centered")

st.title("🧠 Gauss-Jordan AI – Soluționare ecuații liniare")
st.markdown("Introduceți coeficienții pentru o ecuație simplă de forma **a · x = b**")

# Inputuri pentru utilizator
a = st.number_input("Coeficient a", value=3.0)
x_real = st.number_input("Valoare reală a necunoscutei x (pentru test)", value=5.0)

# Buton de rulare
if st.button("🔍 Rezolvă ecuația"):
    model = faza1_train_model()
    st.success("Modelul este antrenat. Afișăm demonstrația mai jos:")
    faza1_demo(model, a, x_real)
