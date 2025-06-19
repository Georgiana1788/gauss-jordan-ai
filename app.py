import streamlit as st
import torch
import torch.nn as nn
import sympy as sp
import random

st.set_page_config(page_title="Gauss-Jordan AI", layout="centered")

# === CALCULATOR RADICALI ===
st.markdown("### 🧮 Calculator de radicali (pătrați, cubici, expresii simbolice)")

expr_input = st.text_input("Introduceți o expresie cu radicali sau necunoscute (ex: sqrt(5), sqrt(x**2 + 1), root(27, 3))", "sqrt(5)")

if expr_input:
    try:
        expr = sp.sympify(expr_input)
        simplified = sp.simplify(expr)
        evaluated = expr.evalf()
        st.write("**Forma simbolică simplificată:**", simplified)
        st.write("**Valoare numerică aproximativă:**", evaluated)
    except Exception as e:
        st.error(f"Eroare la interpretarea expresiei: {e}")

st.markdown("---")

@st.cache_resource
def faza1_train_model():
    data = []
    for _ in range(1000):
        a = random.uniform(1, 10)
        x_true = random.uniform(1, 20)
        b = a * x_true
        data.append(([a, b], [x_true]))
    X = torch.tensor([i[0] for i in data], dtype=torch.float32)
    y = torch.tensor([i[1] for i in data], dtype=torch.float32)

    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(300):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

@st.cache_resource
def faza2_train_model():
    data = []
    for _ in range(2000):
        x = random.uniform(1, 10)
        y = random.uniform(1, 10)
        a = random.uniform(1, 5)
        b = random.uniform(1, 5)
        c = a*x + b*y
        data.append(([a, b, c], [x, y]))
    X = torch.tensor([i[0] for i in data], dtype=torch.float32)
    y = torch.tensor([i[1] for i in data], dtype=torch.float32)

    model = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 2))
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(500):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def faza1_demo(model, a, x_true):
    b = a * x_true
    input_tensor = torch.tensor([[a, b]], dtype=torch.float32)
    x_pred = model(input_tensor).item()
    st.code(f"Ecuația: {a:.2f}x = {b:.2f}\nPasul 1: Împărțim ambele părți la {a:.2f}\nx = {b:.2f} / {a:.2f} = {b/a:.2f}\nRețea prezis: x ≈ {x_pred:.2f}, Eroare: {abs(x_pred - x_true):.4f}")

def faza2_demo(model, a, b, x_true, y_true):
    c = a*x_true + b*y_true
    input_tensor = torch.tensor([[a, b, c]], dtype=torch.float32)
    x_pred, y_pred = model(input_tensor)[0].tolist()
    st.code(f"Ecuația: {a:.2f}x + {b:.2f}y = {c:.2f}\nRezultat real: x = {x_true:.2f}, y = {y_true:.2f}\nRețea prezis: x ≈ {x_pred:.2f}, y ≈ {y_pred:.2f}")

def gauss_jordan_3x3(a1,b1,c1,d1, a2,b2,c2,d2, a3,b3,c3,d3):
    M = sp.Matrix([[a1,b1,c1,d1],[a2,b2,c2,d2],[a3,b3,c3,d3]])
    steps = [f"Matrice inițială:\n{sp.pretty(M)}"]
    M_red = M.rref()[0]
    steps.append(f"Matrice redusă (Gauss-Jordan):\n{sp.pretty(M_red)}")
    x, y, z = M_red[:,3]
    return x, y, z, steps

lang = st.selectbox("🌐 Limba / Language", ["Română", "English"])
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

if phase == TEXT["phases"][0]:
    a = st.number_input(TEXT["a"], value=3.0)
    x_real = st.number_input(TEXT["x_real"], value=5.0)
    if st.button(TEXT["btn"]):
        model = faza1_train_model()
        st.success(TEXT["success"])
        faza1_demo(model, a, x_real)

elif phase == TEXT["phases"][1]:
    a = st.number_input(TEXT["a"], value=2.0)
    b = st.number_input(TEXT["b"], value=3.0)
    x_real = st.number_input(TEXT["x_real"], value=4.0)
    y_real = st.number_input(TEXT["y_real"], value=5.0)
    if st.button(TEXT["btn"]):
        model = faza2_train_model()
        st.success(TEXT["success"])
        faza2_demo(model, a, b, x_real, y_real)

elif phase == TEXT["phases"][2]:
    st.markdown("### ✍️ Introduceți coeficienții sistemului:")
    col1, col2, col3, col4 = st.columns(4)
    with col1: a1 = st.number_input("a1", value=1.0)
    with col2: b1 = st.number_input("b1", value=1.0)
    with col3: c1 = st.number_input("c1", value=1.0)
    with col4: d1 = st.number_input("d1", value=6.0)
    with col1: a2 = st.number_input("a2", value=0.0)
    with col2: b2 = st.number_input("b2", value=2.0)
    with col3: c2 = st.number_input("c2", value=5.0)
    with col4: d2 = st.number_input("d2", value=-4.0)
    with col1: a3 = st.number_input("a3", value=2.0)
    with col2: b3 = st.number_input("b3", value=5.0)
    with col3: c3 = st.number_input("c3", value=-1.0)
    with col4: d3 = st.number_input("d3", value=27.0)

    if st.button(TEXT["btn"]):
        x, y, z, steps = gauss_jordan_3x3(a1,b1,c1,d1,a2,b2,c2,d2,a3,b3,c3,d3)
        st.success("✅ Soluția exactă:")
        st.write(f"x = {x}, y = {y}, z = {z}")
        st.markdown("### 📜 Pași:")
        for s in steps:
            st.code(s)