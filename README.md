# 🧠 Gauss-Jordan AI – Soluționare ecuații liniare & Calculator de radicali

Această aplicație interactivă creată cu **Streamlit** combină puterea rețelelor neuronale cu metode matematice exacte pentru a rezolva ecuații liniare, sisteme de ecuații și expresii cu radicali.

---

## 🔍 Funcționalități principale

### 🔢 Calculator de radicali (pătrați, cubici, simbolici)
- Acceptă expresii precum `sqrt(5)`, `sqrt(x**2 + 1)`, `∛27`
- Afișează rezultatul simbolic și numeric
- Bazat pe biblioteca `sympy`

---

### 🤖 Faza 1 – Rezolvare AI: `ax = b`
- Utilizatorul introduce `a` și `x` real
- Modelul neuronal antrenat prezice `x`
- Afișează demonstrația pas cu pas

### 🧮 Faza 2 – Rezolvare AI: `ax + by = c`
- Utilizatorul introduce `a`, `b`, `x`, `y`
- Modelul neuronal estimează soluția
- Se explică logica și eroarea

### 🧠 Faza 3 – Rezolvare logică exactă: Sistem 3x3
- Utilizatorul introduce toți coeficienții
- Se aplică metoda Gauss-Jordan
- Se afișează **toți pașii de calcul** și soluțiile `x, y, z`

---

## 🗣️ Multilingv
- Aplicația este disponibilă în **Română** și **Engleză**

## ⚡ Performanță
- Modelele AI folosesc `@st.cache_resource` pentru antrenare rapidă și eficientă

---

## 📁 Cum rulezi aplicația
```bash
streamlit run app.py
```

sau o poți accesa direct online prin [Streamlit Cloud](https://streamlit.io/cloud).

---

## 📚 Tehnologii folosite
- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)
- [SymPy](https://www.sympy.org/)
- [ReportLab](https://www.reportlab.com/) (pentru export PDF)

---

## 🧑‍🏫 Creat pentru scopuri educaționale

Proiectul își propune să ajute elevii, studenții și pasionații de matematică să înțeleagă:
- logica rezolvării ecuațiilor liniare
- cum funcționează rețelele neuronale
- cum se aplică metoda Gauss-Jordan pas cu pas