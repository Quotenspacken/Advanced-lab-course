import numpy as np 
import matplotlib.pyplot as plt
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)

def gerade(x, m, b):
    return m*x+b

dt, N = np.genfromtxt('daten/verz.txt', unpack=True)

#fit 
N_cut1  = N[0:19]
dt_cut1 =dt[0:19]
N_cut2  = N[18:24]
dt_cut2 =dt[18:24]
N_cut3  = N[23:40]
dt_cut3 =dt[23:40]

print(dt_cut2[0])
print(dt_cut2[-1])

params1, covariance_matrix1 = np.polyfit(dt_cut1, N_cut1, deg=1, cov=True)
uncertainties1 = np.sqrt(np.diag(covariance_matrix1))
params2, covariance_matrix2 = np.polyfit(dt_cut2, N_cut2, deg=0, cov=True)
uncertainties2 = np.sqrt(np.diag(covariance_matrix2))
params3, covariance_matrix3 = np.polyfit(dt_cut3, N_cut3, deg=1, cov=True)
uncertainties3 = np.sqrt(np.diag(covariance_matrix3))

#Parameter
print("\nRegressionsparameter für Justage")
print("\nfür linke Gerade")
errors1 = np.sqrt(np.diag(covariance_matrix1))
for name, value, error in zip('ab', params1, errors1):
    print(f'{name} = {value:.4f} ± {error:.4f}')
    
print("\nfür rechte Gerade")
errors3 = np.sqrt(np.diag(covariance_matrix3))
for name, value, error in zip('ab', params3, errors3):
    print(f'{name} = {value:.4f} ± {error:.4f}')

print("\nfür Platau")
errors2 = np.sqrt(np.diag(covariance_matrix2))
for name, value, error in zip('ab', params2, errors2):
    print(f'{name} = {value:.4f} ± {error:.4f}')

halfmax = params2[0]/2
params1_err= unp.uarray(params1,errors1)
params2_err= unp.uarray(params2,errors2)
params3_err= unp.uarray(params3,errors3)
h1 = (halfmax-params1_err[1])/params1_err[0]
h3 = (halfmax-params3_err[1])/params3_err[0]

x1 = np.linspace(np.min(dt_cut1), np.max(dt_cut1))
x2 = np.linspace(np.min(dt_cut2), np.max(dt_cut2))
y2 = np.ones(len(x2))
x3 = np.linspace(np.min(dt_cut3), np.max(dt_cut3))
xhalf = np.linspace(noms(h1), noms(h3))
yhalf = np.ones(len(xhalf))
print(f"h1={h1}")
print(f"h3={h3}")
print(f"Halbwertsbreite={h3-h1}")
plt.plot(x1, params1[0]*x1+params1[1], "orange", linewidth=1, label="Regressionsgeraden")
plt.plot(x2, params2[0]*y2, "darkgreen", linewidth=1, label="Plateau")
plt.plot(xhalf, halfmax*yhalf, "r--", linewidth=1, label="halber Plateauwert")
plt.plot(x3, params3[0]*x3+params3[1], "orange", linewidth=1)
plt.errorbar(dt, N, xerr=0, yerr=np.sqrt(N), color="darkblue", ecolor="royalblue", fmt='.', label="Messwerte")
plt.xlabel(r"$\Delta t[ns]$")
plt.ylabel(r"$N [1/s]$")
plt.legend(loc='best')
plt.tight_layout()
plt.grid()
plt.savefig('plots/verz.pdf')

print(f"Plateauwert=({np.mean(N_cut2):.4}+/-{np.std(N_cut2):.3})")