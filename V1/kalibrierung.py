import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
import uncertainties.unumpy as unp


def gerade(x, m, b):
    return m * x + b

t, K = np.genfromtxt("daten/kalibrierung.txt", unpack=True)

params, covariance_matrix = np.polyfit(K, t, deg=1, cov=True)
uncertainties = np.sqrt(np.diag(covariance_matrix))

#Parameter
print("\nGeradenparameter")
errors = np.sqrt(np.diag(covariance_matrix))
for name, value, error in zip("ab", params, errors):
    print(f"{name} = ({value*10**3:.4f} ± {error*10**3:.4f})ns")

x = np.linspace(0, 512)
plt.plot(x, gerade(x, *params), "k", linewidth=1, label="Ausgleichsgerade")
plt.plot(K, t, "r+", markersize=10, label="Messdaten")
plt.xlabel(r"$Kanal$")
plt.ylabel(r"$\Delta t [\mu s]$")
plt.legend(loc="best")
plt.tight_layout()
plt.grid()
plt.savefig('plots/kalibrierung.pdf')