import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as sdevs
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime
from scipy.signal import find_peaks
from uncertainties import ufloat


def lin(x, m, b):
    return m * x + b

def power_law(E, a, b):
    return a * E**b

def gauss(x,sigma,h,a,mu):
    return a+h*np.exp(-((x-mu)/sigma)**2)

def potenz(x,a,b,c,e):
    return a*(x-b)**e+c

def plot_spectrum(data, title, peaks=None, m=1, b=0):
    
    x_channels = np.linspace(0, len(data), len(data))
    
    plt.bar(m *x_channels + b, data)  
    if peaks is not None:
        plt.plot(m*x_channels[peaks]+b, data[peaks], "x", color='red')
    plt.title(title)
    plt.xlabel(r'Energie $E$ (keV)')
    plt.ylabel(r'Zählrate $N$')
    plt.savefig(f'V18/build/{title}.pdf')
    plt.yscale('log')
    plt.savefig(f'V18/build/{title}_log.pdf')
    plt.clf()

def subtract_background(eu_counts, bg_counts, t_eu, t_bg):
    scale = t_eu / t_bg
    # Hintergrund skalieren
    bg_scaled = bg_counts * scale
    # Subtraktion
    corrected_counts = eu_counts - bg_scaled
    return corrected_counts

def gaussian_fit_peaks(test_ind,data):
    peak_inhalt = []
    index_fit = []
    hoehe = []
    unter = []
    sigma = []
    for i in test_ind:
        a=i-40
        b=i+40


        params_gauss_d,covariance_gauss_d=curve_fit(gauss,np.arange(a,b+1),data[a:b+1],p0=[1,data[i],0,i-1])
        errors_gauss_d = np.sqrt(np.diag(covariance_gauss_d))

        sigma_fit=ufloat(params_gauss_d[0],errors_gauss_d[0])
        h_fit=ufloat(params_gauss_d[1],errors_gauss_d[1])
        a_fit=ufloat(params_gauss_d[2],errors_gauss_d[2])
        mu_fit=ufloat(params_gauss_d[3],errors_gauss_d[3])
        #print(h_fit*sigma_fit*np.sqrt(2*np.pi))
        #if i == 3316:
        #    plt.plot(np.arange(a, b+1), data_d[a:b+1], label='Daten')
        #    plt.plot(np.arange(a, b+1), gauss(np.arange(a, b+1), *params_gauss_d), label='Fit')
        #    plt.savefig('build/test.pdf')
        #    plt.clf()
        index_fit.append(mu_fit)
        hoehe.append(h_fit)
        unter.append(a_fit)
        sigma.append(sigma_fit)
        peak_inhalt.append(h_fit*sigma_fit*np.sqrt(2*np.pi))
    return index_fit, peak_inhalt, hoehe, unter, sigma



###########################
### Background spectrum ###
############################
bkg = np.genfromtxt('V18/data/bkg.txt', unpack=True)
plot_spectrum(bkg, 'Hintergrund_Spektrum')

#############################
### Spektrum von Europium ###
##############################
print("Europium-Spektrum wird analysiert...")
Eu = np.genfromtxt('V18/data/Eu.txt', unpack=True)
Eu_channels = np.linspace(0, len(Eu), len(Eu))
E, I= np.genfromtxt('V18/data/Eu_ideal.txt', unpack=True)

Eu_bereinigt = Eu.copy()
Eu_bereinigt[:1000] = 0
Eu_bereinigt[1050:2000] = 0
Eu_bereinigt[-185:] = 0

peaks, _ = find_peaks(Eu_bereinigt, height=np.max(Eu_bereinigt)*0.03, distance=200)  
print(peaks)


plt.bar(Eu_channels, subtract_background(Eu, bkg, 7200, 86400))  
plt.plot(Eu_channels[peaks], Eu_bereinigt[peaks], "x", color='red')                                           
plt.title('Europium-Spektrum')
plt.xlabel(r'Bin-Nummer')
plt.ylabel(r'Zählrate $N$')
plt.savefig(f'V18/build/Europium_Spektrum.pdf')
plt.yscale('log')
plt.savefig(f'V18/build/Europium_Spektrum_log.pdf')
plt.clf()

### Kalibrierung der Energie ###
m,b = np.polyfit(peaks, E, 1)
print(f'Kalibrierung: E = {m:.4f} * Bin + {b:.4f}')

plt.bar(m*Eu_channels+b, subtract_background(Eu, bkg, 7200, 86400))
plt.xticks(m*peaks+b, rotation=45)                                             
plt.title('Europium-Spektrum')
plt.xlabel(r'Energie $E$ (keV)')
plt.ylabel(r'Zählrate $N$')
plt.savefig(f'V18/build/Europium_Spektrum_konfiguriert.pdf')
plt.yscale('log')
plt.savefig(f'V18/build/Europium_Spektrum_konfiguriert_log.pdf')
plt.clf()

### Kalibrierung der zerfalls wahrscheinlichkeiten ###
#Berechnung der Aktivität am Messtag
A0 = 4130  # Bq
t0 = datetime(2000, 10, 1)
t_measure = datetime(2026, 2, 2)  # Beispiel Messdatum

half_life_Eu = 4930 * 24 * 3600  # ¹⁵²Eu Halbwertszeit in Sekunden (ca. 13.537 Jahre)
decay_constant = np.log(2) / half_life_Eu

t_seconds = (t_measure - t0).total_seconds()
A_measure = A0 * np.exp(-decay_constant * t_seconds)
print(f"Aktivität am Messzeitpunkt: {A_measure:.2f} Bq")

# Raumwinkel 
measurement_time = 3600  # Messdauer in Sekunden
distance = 6.91 + 1.5  # Abstand in cm
detector_radius = 2.25   #Detektorradius in cm
omega = 2 * np.pi * (1 - distance / np.sqrt(distance**2 + detector_radius**2))
print(f"Raumwinkel: {omega:.4f} sr")

efficiency = (4*np.pi*Eu[peaks]) / (omega * A_measure * measurement_time * I)

popt, _ = curve_fit(power_law, E, efficiency)

plt.scatter(E, efficiency, label='Messwerte')
E_fit = np.linspace(min(E), max(E), 100)
plt.plot(E_fit, power_law(E_fit, *popt), color='red', label=f'Fit: a={popt[0]:.2e}, b={popt[1]:.2f}')
plt.xlabel('Energie (keV)')
plt.ylabel('Q(E)')
plt.legend()
plt.grid()
plt.savefig('V18/build/Effizienz.pdf')
plt.clf()


#################
###CS-Spektrum###
##################
print("Cs-Spektrum wird analysiert...")
Cs = np.genfromtxt('V18/data/Cs.txt', unpack=True)
Cs = subtract_background(Cs, bkg, 7200, 86400)

Cs_bereinigt = Cs.copy()
Cs_bereinigt[100:1600] = 0
peaks, _ = find_peaks(Cs_bereinigt, height=np.max(Cs)*0.04, distance=200)


plot_spectrum(Cs, 'Cs_Spektrum', peaks=peaks, m=m, b=b)


#Vergleiche zwischen gemessenen und theoretischen Werten der Peaks
e_photo = 661.59
m_e = 511

e_compton = m*peaks[2] + b
e_rueck = m*peaks[1] + b

e_compton_theo = 2*e_photo**2/(m_e*(1+2*e_photo/m_e))
vgl_compton = 1-e_compton/e_compton_theo
print(f'Ein Vergleich des theoretischen E_compton {e_compton_theo} mit dem gemessenen E_compton {e_compton}, beträgt: {vgl_compton} ')

e_rueck_theo = e_photo/(1+2*e_photo/m_e)
vgl_rueck = 1-e_rueck/e_rueck_theo
print(f'Ein Vergleich des theoretischen E_rueck {e_rueck_theo} mit dem gemessenen E_rueck {e_rueck}, beträgt: {vgl_rueck} ')

#Führe wieder Gausß-Fit für den Vollenergiepeak durch, um Peakhöhe bestimmen zu können
a=peaks[3].astype('int')-50
c=peaks[3].astype('int')+50

params = m,b

params_gauss_b,covariance_gauss_b=curve_fit(gauss,lin(np.arange(a,c+1),*params),Cs[a:c+1],p0=[1,Cs[peaks[3]],0,lin(peaks[3]-0.1,*params)])
errors_gauss_b = np.sqrt(np.diag(covariance_gauss_b))


#Fasse Wert und Ungenauigkeit der Fit-Parameter wieder jeweils zusammen
sigma_fit=ufloat(params_gauss_b[0],errors_gauss_b[0])
h_fit=ufloat(params_gauss_b[1],errors_gauss_b[1])
a_fit=ufloat(params_gauss_b[2],errors_gauss_b[2])
mu_fit=ufloat(params_gauss_b[3],errors_gauss_b[3])

Z_photo=h_fit*sigma_fit*np.sqrt(2*np.pi)
print(f'Die Rate des Vollenergiepeaks liegt bei {Z_photo} keV.')

Z_3 = Cs[a:c+1]*sigma_fit*np.sqrt(2*np.pi)
plt.plot(lin(np.arange(a,c+1),*params), gauss(lin(np.arange(a,c+1),*params), *params_gauss_b)*noms(sigma_fit)*np.sqrt(2*np.pi), 'k-', label='Fit')
plt.errorbar(lin(np.arange(a,c+1),*params), noms(Z_3), yerr=sdevs(Z_3), fillstyle= None, fmt=' x', label='Daten')
plt.axhline(y=0.5*Cs[peaks[3]]*noms(sigma_fit)*np.sqrt(2*np.pi), xmin = 0.31, xmax = 0.675, color='g',linestyle='dashed', label=' half-value width')
plt.axhline(y=0.1*Cs[peaks[3]]*noms(sigma_fit)*np.sqrt(2*np.pi), xmin = 0.158, xmax = 0.81,color='r',linestyle='dashed', label='tenth-value width')
plt.ylabel('Peakhöhe $Z$')
plt.xlabel('Energie $E$/keV')
plt.legend(loc='best')
plt.grid()
plt.savefig('V18/build/Cs_full_peak.pdf')
plt.clf()

print('\nVergleich Halb- zu Zehntelwertsbreite:')
#lin beschreibt noch die lineare Regression vom beginn der Auswertung
h_g = ufloat(2.2, 0.2)
print('Halbwertsbreite Gemessen: ', h_g)
h_t = np.sqrt(8*np.log(2))*sigma_fit
print('Halbwertsbreite Theorie: ', h_t)
z_g = ufloat(4.0, 0.3)
print('Zehntelbreite Gemessen: ', z_g)
z_t = np.sqrt(8*np.log(10))*sigma_fit
print('Zehntelbreite Theorie: ', z_t)

print('Verhältnis der Halbwertsbreiten Werte: ', (h_g-h_t)/h_t, 'und der Zehntelbreiten: ', (z_g-z_t)/z_t)
print('Verhältnis zwischen gemessener', z_g/h_g, '', z_t/h_t,'\n')


###Plotte das zugeordnete Cs-Spektrum und setze Horizontale bei Zehntel- und Harlbwertsbreite
x=np.linspace(1,8192,8192)
plt.plot(lin(x, *params), Cs,'r-',label='Fit')
plt.plot(lin(peaks, *params),Cs[peaks],'bx',label='Peaks')
plt.axhline(y=0.5*Cs[peaks[3]], color='g',linestyle='dashed', label='Halbwertshöhe')
print('Halbwertshöhe', 0.5*Cs[peaks[3]])
print('Zehntelwertshöhe', 0.1*Cs[peaks[3]])
plt.axhline(y=0.1*Cs[peaks[3]], color='r',linestyle='dashed', label='Zehntelhöhe')
plt.xlim(0,700)
plt.xlabel(r'E$_\gamma\:/\: \mathrm{keV}$')
plt.ylabel(r'Einträge')
plt.grid()
plt.legend(loc='upper left')
plt.savefig('V18/build/Cs.pdf')
plt.yscale('log')
plt.savefig('V18/build/Cs_log.pdf')
plt.clf()

inhalt_photo = ufloat(sum(Cs[1649-13:1649+9]*noms(sigma_fit)*np.sqrt(2*np.pi)), sum(np.sqrt(Cs[1649-13:1649+9]*noms(sigma_fit)*np.sqrt(2*np.pi))))
print('\nInhalt des Photo-Peaks: ', inhalt_photo)

min_ind_comp = 53
inhalt_comp = ufloat(sum(Cs[min_ind_comp:peaks[3]]*noms(sigma_fit)*np.sqrt(2*np.pi)), sum(np.sqrt(Cs[min_ind_comp:peaks[3]]*noms(sigma_fit)*np.sqrt(2*np.pi))))
print(f'Der Inhalt des Compton-Kontinuums, liegt bei: {inhalt_comp}')

mu_ph = ufloat(0.007, 0.003) #in cm^-1
mu_comp = ufloat(0.35, 0.07)
l=3.9
abs_wahrsch_ph = 1-unp.exp(-mu_ph*l)
abs_wahrsch_comp = 1-unp.exp(-mu_comp*l)
print(f'Die absolute Wahrscheinlichkeit eine Vollenergiepeaks liegt bei: {abs_wahrsch_ph} Prozent')
print(f'Die absolute Wahrscheinlichkeit eine Comptonpeaks liegt bei: {abs_wahrsch_comp} Prozent\n')



##########################
### Spektrum von Barium###
###########################
print("Barium-Spektrum wird analysiert...")

Ba = np.genfromtxt('V18/data/Ba.txt', unpack=True)
Ba_bereinigt = Ba.copy()
Ba_bereinigt[4000:8000] = 0

peaks, _ = find_peaks(Ba_bereinigt, height=np.max(Cs)*0.05, distance=20)

plot_spectrum(Ba_bereinigt, 'Ba Spectrum',peaks=peaks, m=m, b=b)

A=ufloat(4130,60) #Aktivität Europium-Quelle am 01.10.2000
t_halb = ufloat(4943,5) #Halbwertszeit Europium in Tagen
dt = 25*365.25 + 92 #Zeitintervall in Tagen

A_jetzt=A*unp.exp(-unp.log(2)*dt/t_halb)#Aktivität Versuchstag

E_ba, W_ba, peaks_ind_ba, E_ba_echt= np.genfromtxt('V18/data/Ba_ideal.txt', unpack=True)
index_ba, peakinhalt_ba, hoehe_ba, unter_ba, sigma_ba = gaussian_fit_peaks(peaks_ind_ba.astype('int'), Ba_bereinigt)

index_f, peakinhalt, hoehe, unter, sigma = gaussian_fit_peaks(peaks_ind_ba.astype('int'),Ba)
omega_4pi = (1-a/(a**2+0.0225**2)**(0.5))/2
E_ba_det = []
for i in range(len(index_ba)):
    E_ba_det.append(lin(index_ba[i],*params))
Q=[]
Z=[]
for i in range(len(W_ba)):
    Z.append(np.sqrt(2*np.pi)*hoehe_ba[i]*sigma_ba[i])
    Q.append(Z[i]/(omega_4pi*A_jetzt*W_ba[i]/100*3600))

params2, covariance2= curve_fit(potenz,noms(E_ba_det),noms(Q),sigma=sdevs(Q), p0=[1, 0.1, 0, 0.5])

A=peakinhalt_ba[1:]/(3600*omega_4pi*W_ba[1:]/100*potenz(E_ba_det[1:],*params2)) #nur die mit E>150keV mitnehmen

A_det = []

A_det.append(ufloat(0, 0))

for i in A:
    A_det.append(i)


#berechnung der neuen Effizienz
Z_d = [ufloat(0, 0)]
Q_d = [ufloat(0, 0)]
for i in range(1, len(W_ba)):
    Z_d.append(np.sqrt(2*np.pi)*hoehe_ba[i]*sigma_ba[i])
    Q_d.append(Z[i]/(omega_4pi*A_det[i]*W_ba[i]/100*3600))

print('Berechnete Aktivität der betrachteten Emissionslinien mit dazu korrespondierenden Detektor-Effizienzen:')
print('W%: ', W_ba)
print('Q: ', Q_d)
print('Z: ', Z_d)       
print('E: ', E_ba_det)
print('A: ', A_det)

A_gem = ufloat(np.mean(noms(A)),np.mean(sdevs(A)))
print('gemittelte Aktivität',A_gem)



#########################################
### Spektrum von Unbekanntem Material ###
#########################################
print("Unbekanntes Spektrum wird analysiert...")

Ukn = np.genfromtxt('V18/data/Unknown.txt', unpack=True)
Ukn_bereinigt = Ukn.copy()

peaks, _ = find_peaks(Ukn_bereinigt, height=np.max(Ukn)*0.30, distance=20)
plot_spectrum(subtract_background(Ukn, bkg, 7200, 86400), 'Unknown Spectrum',peaks=peaks, m=m, b=b)



E_e, W_e, peaks_ind_e = np.genfromtxt('V18/data/Co_ideal.txt', unpack=True)
index_e, peakinhalt_e, hoehe_e, unter_e, sigma_e = gaussian_fit_peaks(peaks_ind_e.astype('int'),Ukn)
print(f'Peakinhalt {peakinhalt_e} Hoehe {hoehe_e}, Sigma {sigma_e}')

E_e_det = []
for i in range(len(W_e)):
    E_e_det.append(lin(index_e[i], *params))


Z_e=[]
Q_e = []
A_e=[]
for i in range(0, len(W_e)):
    Z_e.append(np.sqrt(2*np.pi)*hoehe_e[i]*sigma_e[i])
    A_e.append(Z_e[i]/(3600*omega_4pi*W_e[i]/100*potenz(E_e_det[i],*params2)))
    Q_e.append(Z[i]/(omega_4pi*A_e[i]*W_e[i]/100*3600))

print(f'\nDaten zur Berechnung der Akivität: {E_e}, {params2}, den Peakinhalt Z {Z_e}, die Effizienz Q {Q_e} und der Aktivität {A_e}')
print('gemittelte Aktivität für Cobalt: ', np.mean(A_e))
print('Berechnete Aktivität der betrachteten Emissionslinien mit dazu korrespondierenden Detektor-Effizienzen:')
print('W%: ', W_e)
print('Q: ', Q_e)
print('Z: ', Z_e)
print('E: ', E_e)   
print('A: ', A_e)