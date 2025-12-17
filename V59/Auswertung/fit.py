import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from numpy.polynomial import Polynomial

# --- 1️⃣ CSV einlesen ---
df = pd.read_csv("scope_4.csv", skiprows=1)  # zwei Header-Zeilen überspringen [scope_4.csv]
x = df["second"].values
y = df["Volt"].values

# --- 2️⃣ Lokale Maxima finden ---
peaks_max, _ = find_peaks(y, height=0.005)
peaks_min, _ = find_peaks(-y)

x_max = x[peaks_max]
y_max = y[peaks_max]
x_min = x[peaks_min]
y_min = y[peaks_min]

# Optional: nur einen bestimmten Bereich verwenden
#mask_range = (x_max >= 0) & (x_max <= 10e-6)
#x_sel_max, y_sel_max = x_max[mask_range], y_max[mask_range]

#mask_range_min = (x_min >= 0) & (x_min <= 10e-6)
#x_sel_min, y_sel_min = x_min[mask_range_min], y_min[mask_range_min]

# --- 3️⃣ Polynom-Fits für obere und untere Hüllkurve ---
deg_upper, deg_lower = 6, 6   # Grad des Fits anpassen bei Bedarf
poly_upper_coeffs = np.polyfit(x_max, y_max, deg_upper)
poly_lower_coeffs = np.polyfit(x_min, y_min, deg_lower)
#poly_upper_coeffs = np.polyfit(x_sel_max, y_sel_max, deg_upper)
#poly_lower_coeffs = np.polyfit(x_sel_min, y_sel_min, deg_lower)


poly_upper = np.poly1d(poly_upper_coeffs)
poly_lower = np.poly1d(poly_lower_coeffs)

# Fit-Kurven berechnen [scope_4.csv]
x_fit = np.linspace(min(x), max(x), 1000)
y_fit_upper = poly_upper(x_fit)
y_fit_lower = poly_lower(x_fit)

# --- 4️⃣ Modulationsgrad berechnen ---
mask_time = x_fit >= -3e-6

Amax_global = np.max(y_fit_upper[mask_time])
Amin_global = np.min(y_fit_lower[mask_time])

m_modulation_index = (Amax_global - Amin_global) / (Amax_global + Amin_global)

print(f"Maximale Amplitude: {Amax_global:.5f} V")
print(f"Minimale Amplitude: {Amin_global:.5f} V")
print(f"Modulationsgrad m: {m_modulation_index:.3f}")

# --- 5️⃣ Plot zur Kontrolle ---
plt.figure(figsize=(10,5))
plt.plot(x*1e6, y*1e3, label="Messdaten", color="gray", alpha=0.5)
plt.plot(x_fit*1e6, y_fit_upper*1e3,
         color="red", linewidth=2,
         label="Obere Hüllkurve")
plt.plot(x_fit*1e6, y_fit_lower*1e3,
         color="blue", linewidth=2,
         label="Untere Hüllkurve")
plt.xlabel("Zeit [µs]")
plt.ylabel("Spannung [mV]")
plt.title(f"AM-Hüllkurven und Modulationsgrad m={m_modulation_index:.3f}")
plt.legend()
plt.grid(True)
plt.show()