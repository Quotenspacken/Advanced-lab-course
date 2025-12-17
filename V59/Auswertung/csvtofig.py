import pandas as pd
import matplotlib.pyplot as plt

# CSV-Datei einlesen
df = pd.read_csv("scope_8.csv", skiprows=1)
#print(df.columns)
# Beispielsweise zwei Spalten plotten:
#plt.figure(figsize=(8, 5))
#plt.plot(df['Zeit'], df["Temperatur"], marker="o")
plt.plot(df["second"], df["Volt"])
#plt.plot(df["second"], df["Volt.1"])
#plt.plot(df["second"], df["Volt"], df["Volt.1"])
plt.title("Spannungsverlauf über die Zeit")
plt.xlabel("[s]")
plt.ylabel("[V]")
plt.grid(True)
plt.savefig("csv8.png")
plt.show()