import numpy as np 
import matplotlib.pyplot as plt
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit

U=0#U=0.134
def ef(x, N0, l):
    return N0*np.exp(-l*x)+U

N = np.genfromtxt('daten/messung_jal_rue.Spe', skip_header=12, skip_footer=15, unpack=True)
K = np.linspace(1,len(N),len(N))

#Kanäle in Zeit umrechnen
a = ufloat(0.0217391, 0.0)
b = ufloat(0.1521739, 0.0)
t = a * K + b

#Lösche unphysikalische Werte (N=0)
indexes=np.zeros(len(N))
n=0
for i in range(len(N)):
    if N[i]<1:
        indexes[n]=i
        n=n+1
indexes = indexes.astype(int)

N_cut = np.delete(N, indexes)
t_cut = np.delete(t, indexes)

#Regression
#params, cov = curve_fit(ef,  noms(t_cut),  N_cut)
params, cov = curve_fit(ef,  noms(t[4:-286]),  N[4:-286])

print("\nRegressionsparameter für die Lebensdauer sind")
errors = np.sqrt(np.diag(cov))
for name, value, error in zip('Nl', params, errors):
    print(f'{name} = {value:.8f} ± {error:.8f}')

#Plot
x=np.linspace(0, 6)
#plt.errorbar(noms(t),     N,     xerr=stds(t),     yerr=np.sqrt(N),     color='red', ecolor='grey',  markersize=3.5, elinewidth=0.5, fmt='.', label="entfernte Daten")
plt.errorbar(noms(t[4:-286]), N[4:-286], xerr=stds(t[4:-286]), yerr=np.sqrt(N[4:-286]), color='navy', ecolor='grey', markersize=3.5, elinewidth=0.5, fmt='.', label="Messdaten")
plt.plot(x, ef(x, params[0], params[1]), color='lime', label="Fit")
plt.xlabel(r"$t [\mu s]$")
plt.ylabel(r"$N [1/s]$")
plt.legend(loc='best')
plt.tight_layout()
plt.grid()
plt.savefig('Latex/Abbildungen/myons.pdf')

#Berechnung der mittleren Lebensdauer
lam=ufloat(params[1], errors[1])
tau=1/lam
print(f"\nMittlere Lebensdauer der Myonen tau=({tau:.4})us")

tau_lit=ufloat(2.197,0.0000022)
p=(tau-tau_lit)/tau_lit *100
print(p)