# Explicații educaționale – Radicali și Metoda Gauss-Jordan

---

## Ce sunt radicalii?

Un **radical** reprezintă o operație matematică prin care se caută o valoare care, ridicată la o anumită putere, oferă un rezultat dat.
Cel mai cunoscut este **radicalul pătrat**: √x, adică acel număr care, ridicat la pătrat, dă x.

### Istoric:
- Simbolul radicalului (√) a fost folosit prima dată în 1525 de către **Christoff Rudolff**, un matematician german.
- Cuvântul „radical” vine din latină: *radix* = rădăcină.

### Unde se folosesc:
- În fizică (calculul vitezei, energiei)
- În economie (rădăcini pătrate pentru variații)
- În calculul distanței (teorema lui Pitagora)
- În programare și algoritmi matematici

### Cum se calculează:
- √9 = 3 pentru că 3² = 9
- ∛27 = 3 pentru că 3³ = 27
- √(x² + 2x + 1) = x + 1 (dacă x > 0)

---

## Metoda Gauss-Jordan

### Istoric:
- Numită după **Carl Friedrich Gauss** și **Wilhelm Jordan** (secolul XIX)
- Extinde metoda de eliminare Gauss pentru a duce o matrice în forma „identitate”
- A fost dezvoltată ca o tehnică sistematică de rezolvare a sistemelor liniare

### La ce folosește:
- Se aplică pentru a rezolva **sisteme de ecuații liniare** (cu 2, 3 sau mai multe necunoscute)
- Se folosește și în informatică, rețele electrice, inginerie, statistici

### Cum funcționează:
1. Scrii sistemul sub formă de matrice extinsă
2. Transformi matricea în forma identitate:
   ```
   [1 0 0 | x]
   [0 1 0 | y]
   [0 0 1 | z]
   ```
3. Fiecare linie (L1, L2, L3) se transformă prin:
   - împărțire (pentru a obține pivot 1)
   - scădere de linii (pentru a elimina necunoscutele)

Metoda este logică, pas cu pas, și garantează soluția exactă dacă sistemul este compatibil.

---