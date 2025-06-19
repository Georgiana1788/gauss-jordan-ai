
import streamlit as st
import torch
import torch.nn as nn
import random

st.set_page_config(page_title="Gauss-Jordan AI", layout="centered")

# === Funcții integrate ===

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

def faza1_demo(model, a, x_true):
    b = a * x_true
    input_tensor = torch.tensor([[a, b]], dtype=torch.float32)
    x_pred = model(input_tensor).item()
    st.code(f"""Ecuația: {a:.2f}x = {b:.2f}
Pasul 1: Împărțim ambele părți la {a:.2f}
Pasul 2: x = {b:.2f} / {a:.2f} = {b/a:.2f}
Rezultat real: x = {x_true:.2f}
Rețeaua a prezis: x ≈ {x_pred:.2f} (eroare: {abs(x_pred - x_true):.4f})""")

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

def faza2_demo(model, a, b, x_true, y_true):
    c = a*x_true + b*y_true
    input_tensor = torch.tensor([[a, b, c]], dtype=torch.float32)
    x_pred, y_pred = model(input_tensor)[0].tolist()
    st.code(f"""Ecuația: {a:.2f}x + {b:.2f}y = {c:.2f}
Rezultat real: x = {x_true:.2f}, y = {y_true:.2f}
Rețea prezis: x ≈ {x_pred:.2f}, y ≈ {y_pred:.2f}
Eroare totală: {(abs(x_pred - x_true) + abs(y_pred - y_true)):.4f}""")

def faza3_train_model():
    data = []
    for _ in range(3000):
        x, y, z = [random.uniform(1, 10) for _ in range(3)]
        coeffs = [[random.uniform(1, 5) for _ in range(3)] for _ in range(3)]
        results = [sum(c*v for c, v in zip(row, [x, y, z])) for row in coeffs]
        flat = [n for row in coeffs for n in row]
        data.append((flat + results, [x, y, z]))
    X = torch.tensor([i[0] for i in data], dtype=torch.float32)
    y = torch.tensor([i[1] for i in data], dtype=torch.float32)

    model = nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 3))
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(700):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def faza3_demo(model, x_true, y_true, z_true):
    vars = [x_true, y_true, z_true]
    coeffs = [[random.uniform(1, 5) for _ in range(3)] for _ in range(3)]
    results = [sum(c*v for c, v in zip(row, vars)) for row in coeffs]
    flat_input = [n for row in coeffs for n in row] + results
    input_tensor = torch.tensor([flat_input], dtype=torch.float32)
    x_pred, y_pred, z_pred = model(input_tensor)[0].tolist()
    system = "\n".join(
        [f"{r[0]:.2f}x + {r[1]:.2f}y + {r[2]:.2f}z = {res:.2f}"
         for r, res in zip(coeffs, results)]
    )
    st.code(f"""Sistemul:
{system}
Rezultat real: x = {x_true:.2f}, y = {y_true:.2f}, z = {z_true:.2f}
Rețea prezis: x ≈ {x_pred:.2f}, y ≈ {y_pred:.2f}, z ≈ {z_pred:.2f}
Eroare totală: {(abs(x_pred - x_true) + abs(y_pred - y_true) + abs(z_pred - z_true)):.4f}""")

# === Interfață ===

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
