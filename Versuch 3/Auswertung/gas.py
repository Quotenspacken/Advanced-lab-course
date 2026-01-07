import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import scipy.constants as const
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)

# functions
def n_air_exp(M):
    return M*lamda/(L) + 1

def n_air_theo(p, A, b):
    return (3/2) * A * p/(const.R *T_0) + b

# constants
lamda = 632.99e-9 # wavelength of the laser (meter)
T_0 = 273.15 + 21.7 # Raumtemperatur
L = ufloat(0.1, 0.1e-3)

p, m1, m2, m3 = np.genfromtxt("Versuch 3/data/gas.dat", unpack = True)

M = unp.uarray(np.mean([m1, m2, m3], axis = 0), np.std([m1, m2, m3], axis = 0))

n = n_air_exp(m1)
print("---------------------------------------------------")
print("Brechungsindices Luft:")
for i in range(len(n)):
    print(f"{n[i]:.7f}")
print("---------------------------------------------------")


params2, pcov2 = op.curve_fit(n_air_theo, p, noms(n))   
err2 = np.sqrt(np.diag(pcov2))

n_air_exp = 3/2 * ufloat(params2[0], err2[0])*(1013)/(const.R *(273.15 + 22.5)) + ufloat(params2[1], err2[1])

print("--------------------------------------------------")
print("Fit: Refractive Index of Air:")
print(f"A = {params2[0]:.4e} +- {err2[0]:.4e}")
print(f"b = {params2[1]:.8f} +- {err2[1]:.8f}")
print(f"Experimental Value: n = {n_air_exp}")
print("--------------------------------------------------")

# Plot
x = np.linspace(0, 1000, 100)
fig, ax1 = plt.subplots(figsize=(8, 6))

# Daten und Fit im Hauptplot darstellen
ax1.errorbar(p, noms(n), yerr=stds(n), fmt='.', markersize=5, color='black', ecolor='dimgray', elinewidth=2, capsize=2, label="Data")
ax1.plot(x, n_air_theo(x, *params2), color='dodgerblue', linestyle='-', linewidth=2, label="Fit")

# Beschriftungen und Grenzen für den Hauptplot
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-4, 4))  # Setzen des Bereichs für die wissenschaftliche Notation
ax1.yaxis.set_major_formatter(formatter)
ax1.ticklabel_format(style='sci', axis='y', scilimits=(-4, 4))
ax1.set_ylabel(r"$n$", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.7)

# Verschönern und speichern
plt.tight_layout()
fig.patch.set_facecolor('whitesmoke')
plt.savefig("Versuch 3/Latex/build/n_air.pdf", bbox_inches='tight')
plt.close()