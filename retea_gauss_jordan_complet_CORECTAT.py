
import torch
import torch.nn as nn
import random

# -------- FAZA 1: ax = b --------
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
    print("\n--- FAZA 1 ---")
    print(f"Ecuația: {a:.2f}x = {b:.2f}")
    print(f"Pasul 1: Împărțim ambele părți la {a:.2f}")
    print(f"Pasul 2: x = {b:.2f} / {a:.2f} = {b/a:.2f}")
    print(f"Rețeaua a prezis: x ≈ {x_pred:.2f} (eroare: {abs(x_pred - x_true):.4f})")

# -------- FAZA 2: ax + by = c --------
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
    print("\n--- FAZA 2 ---")
    print(f"Ecuația: {a:.2f}x + {b:.2f}y = {c:.2f}")
    print(f"Pas 1: Alegem a, b și c")
    print(f"Pas 2: Rezolvăm sistemul")
    print(f"Rezultat real: x = {x_true:.2f}, y = {y_true:.2f}")
    print(f"Rețea prezis: x ≈ {x_pred:.2f}, y ≈ {y_pred:.2f}")

# -------- FAZA 3: Sistem cu 3 necunoscute --------
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
    print("\n--- FAZA 3 ---")
    print("Sistemul:")
    for i, row in enumerate(coeffs):
        print(f"{row[0]:.2f}x + {row[1]:.2f}y + {row[2]:.2f}z = {results[i]:.2f}")
    print(f"Rezultat real: x = {x_true:.2f}, y = {y_true:.2f}, z = {z_true:.2f}")
    print(f"Rețea prezis: x ≈ {x_pred:.2f}, y ≈ {y_pred:.2f}, z ≈ {z_pred:.2f}")

# ---------------- EXECUTĂM TOATE FAZELE ----------------
if __name__ == '__main__':
    model1 = faza1_train_model()
    faza1_demo(model1, 3, 5)

    model2 = faza2_train_model()
    faza2_demo(model2, 2, 3, 4, 5)

    model3 = faza3_train_model()
    faza3_demo(model3, 2, 3, 4)
