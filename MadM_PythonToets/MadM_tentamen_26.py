# -*- coding: utf-8 -*-
"""
Created on Mon apr 7 12:21:14 2025

@author: alagerberg and rvdSlikke
this script loads the video and IMU data
"""
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

#%% inladen van de markerdata
markerdata = np.loadtxt("video_data.csv", delimiter=";", dtype=float)
# toe = markerdata[:, :2]
ankle = markerdata[:, 2:4]
knee = markerdata[:, 4:6]
hip = markerdata[:, 6:8]
pelvis = markerdata[:, 8:10]

#%% inladen van de tussenresultaten van de gewrichtshoeken en hoeksnelheid (gebruik die alleen als je zelf niet tot een antwoord komt bij een van de vragen)
#results = np.loadtxt("video_results2.csv", delimiter=";", dtype=float)
#knee_ang_deg = results[:, 0]
#knee_ang_vel = results[:, 1]
#hip_ang_deg = results[:,2]

#%% inladen van de indexen van de maximale knieflexie events. Gebruik die bij vraag 1g als je ze zelf niet berekenen kon bij 1c
#idx_flex_knee = np.loadtxt("knee_idx.csv", delimiter=";").astype(np.int64)

# %% Opdrachten (1a tm g) over het onderdeel 2D-videoanalyse
# in het script staat steeds een korte samenvatting van de opdracht, maar lees vooral ook de complete opdracht in het meegeleverde opdrachtenformulier.
# Run het script voordat je met de vragen begint. Pas daarna zijn de variabelen die hierboven ingeladen worden beschikbaar
# de variabelen die gemaakt worden in regel 21 tm 24 en 27, heb je alleen nodig als je bij
# een van de onderstaande opdrachten niet tot een antwoord kunt komen.
# Haal in dat geval de # weg en run je script opnieuw om verder te kunnen werken met de betreffende variabele.

# %% Opdracht 1a. (0,5 pt)
# Maak een zogenaamde X,Y plot van de markerpaden van de ankle, knee en hip marker (plot de X waarden tegen de Y waarden). 
# Zorg voor een passende grafiek titel en passende titels bij de X en de Y-as van de grafiek. 
plt.figure()
plt.plot(ankle[:, 0], ankle[:, 1], label="Ankle")
plt.plot(knee[:, 0], knee[:, 1], label="Knee")
plt.plot(hip[:, 0], hip[:, 1], label="Hip")
plt.plot(pelvis[:, 0], pelvis[:, 1], label="Pelvis")
plt.title("2D markerpaden tijdens fietsen")
plt.xlabel("X-positie (pixels)")
plt.ylabel("Y-positie (pixels)")
plt.axis("equal")
plt.legend()
plt.grid(True)


# %% Opdracht 1b (1 pt)  
# Gebruik de drie markers (ankle, knee en hip) om de kniehoek (in graden) te berekenen. 
# Maak een plot van de kniehoek (in graden) met passende titels op de X en de Y as.  
vec_knee_to_ankle = ankle - knee
vec_knee_to_hip = hip - knee

dot_prod = np.sum(vec_knee_to_ankle * vec_knee_to_hip, axis=1)
norm_prod = np.linalg.norm(vec_knee_to_ankle, axis=1) * np.linalg.norm(vec_knee_to_hip, axis=1)
cos_knee = dot_prod / norm_prod
cos_knee = np.clip(cos_knee, -1.0, 1.0)
knee_ang_deg = np.degrees(np.arccos(cos_knee))

plt.figure()
plt.plot(knee_ang_deg, label="Kniehoek")
plt.title("Kniehoek tijdens fietsen")
plt.xlabel("Frame")
plt.ylabel("Hoek (graden)")
plt.legend()
plt.grid(True)


# %% Opdracht 1c (1 pt)  
# Bepaal alle indexen (framenummers) van de momenten dat de knie tijdens de diverse cycli in de uiterste gebogen stand is (de dalen in de kniehoek grafiek). 
# Gebruik die indexen om een rode stip te plotten in de grafiek van de kniehoek 
# (maak dus nogmaals een plot van de kniehoek en voeg de rode stippen toe).
idx_flex_knee, _ = find_peaks(-knee_ang_deg, distance=20)

plt.figure()
plt.plot(knee_ang_deg, label="Kniehoek")
plt.plot(idx_flex_knee, knee_ang_deg[idx_flex_knee], "ro", label="Maximale knieflexie")
plt.title("Kniehoek met momenten van maximale flexie")
plt.xlabel("Frame")
plt.ylabel("Hoek (graden)")
plt.legend()
plt.grid(True)


# %% Opdracht 1d (0,5 pt)  
# Bereken de hoeksnelheid in het kniegewricht (in graden per seconde). De framerate van de video was 120 beeldjes per seconde (fps). maak een plot van je resultaat.
fps_video = 120.0
knee_ang_vel = np.gradient(knee_ang_deg) * fps_video

plt.figure()
plt.plot(knee_ang_vel, label="Hoeksnelheid knie")
plt.title("Hoeksnelheid van de knie tijdens fietsen")
plt.xlabel("Frame")
plt.ylabel("Hoeksnelheid (graden/s)")
plt.legend()
plt.grid(True)

# %% Opdracht 1e(1 pt)  
# Schrijf de code om de gemiddelde trapfrequentie te berekenen. Druk die uit in RPM (Rotations Per Minute = omwentelingen per minuut) 
cycle_period_frames = np.diff(idx_flex_knee)
cycle_period_sec = cycle_period_frames / fps_video
mean_cycle_period_sec = np.mean(cycle_period_sec)
cadence_rpm = 60.0 / mean_cycle_period_sec
print(f"Gemiddelde trapfrequentie: {cadence_rpm:.2f} RPM")


# %% Opdracht 1f (0,5 pt) 
# Bereken de heuphoek (in graden) met behulp van de knee, hip en pelvis marker.
# maak een plot van zowel de kniehoek als de heuphoek (in dezelfde plot) zorg ook voor legends en voor passende titels voor de grafiek en op de assen.
vec_hip_to_knee = knee - hip
vec_hip_to_pelvis = pelvis - hip

dot_prod_hip = np.sum(vec_hip_to_knee * vec_hip_to_pelvis, axis=1)
norm_prod_hip = np.linalg.norm(vec_hip_to_knee, axis=1) * np.linalg.norm(vec_hip_to_pelvis, axis=1)
cos_hip = dot_prod_hip / norm_prod_hip
cos_hip = np.clip(cos_hip, -1.0, 1.0)
hip_ang_deg = np.degrees(np.arccos(cos_hip))

plt.figure()
plt.plot(knee_ang_deg, label="Kniehoek")
plt.plot(hip_ang_deg, label="Heuphoek")
plt.title("Kniehoek en heuphoek tijdens fietsen")
plt.xlabel("Frame")
plt.ylabel("Hoek (graden)")
plt.legend()
plt.grid(True)


# %% Opdracht 1g (0,5 pt) 
# Bereken alle tijdsverschillen tussen de uiterste flexiestand van de knie en de uiterste flexiestand van de heup.
# Bereken op basis van die uitkomsten ook het gemiddelde tijdsverschil. 
# Lees de tips in de opdrachtsomschrijving. 
idx_flex_hip, _ = find_peaks(-hip_ang_deg, distance=20)

n_cycles = min(len(idx_flex_knee), len(idx_flex_hip))
frame_diff_flex = idx_flex_hip[:n_cycles] - idx_flex_knee[:n_cycles]
time_diff_flex_sec = frame_diff_flex / fps_video
mean_time_diff_flex_sec = np.mean(time_diff_flex_sec)

print("Tijdsverschillen knie-heup flexie per cyclus (s):", time_diff_flex_sec)
print(f"Gemiddeld tijdsverschil knie-heup flexie: {mean_time_diff_flex_sec:.4f} s")

# %% Vragen over de analyse van IMU data
# Lees de IMU data in en sla de eerste "header" regel over
imu_data = np.loadtxt("sensors.csv", delimiter=",", skiprows=1).astype(float)

# %%
# De imu_data bevat de volgende kolommen:
# Tijdas (s) kolom 0
# Gyro X Y en Z (deg/s), kolom 1, 2 en 3
# Versnelling (m/s^2, Acc) X, Y, en Z, kolom 4, 5 en 6
# Magnetometer X, Y en Z, kolom 7, 8, en 9

# %% Opdracht 2a (0,5 p) Bereken de gemiddelde samplefrequentie op basis van de gehele tijdsas en sla deze op als "fs".
time = imu_data[:, 0]
dt = np.diff(time)
fs = 1.0 / np.mean(dt)
print(f"Gemiddelde samplefrequentie fs: {fs:.2f} Hz")


# %% Opdracht 2b (0,5 p)  
# Bereken de minimale en maximale samplefrequentie die binnen dit signaal optreden 
fs_inst = 1.0 / dt
fs_min = np.min(fs_inst)
fs_max = np.max(fs_inst)
print(f"Minimale samplefrequentie: {fs_min:.2f} Hz")
print(f"Maximale samplefrequentie: {fs_max:.2f} Hz")


#%% Opdracht 2c (0,5 p) 
# Zet het versnellingssignaal van X om naar de hoek (in graden) van de sensor ten opzichte van de verticaal. Sla deze op als “upp_leg_angle_acc_x” 
# Waarschijnlijk wordt er bij het berekenen een waarschuwing gegeven. Leg uit waar deze waarschuwing vandaan komt (zet dit als commentaar bij je code). . 
acc_x = imu_data[:, 4]
g = 9.81
# Waarschuwing komt doordat acc_x/g soms buiten [-1, 1] valt (ruis + dynamische versnelling),
# waardoor arccos een ongeldige invoer krijgt en NaN kan opleveren.
upp_leg_angle_acc_x = np.degrees(np.arccos(acc_x / g))


#%% Opdracht 2d (0,5 p) 
# Plot het berekende hoeksignaal signaal tegen de tijd, maar alleen van sample 200 tot 2000. 
idx_start = 200
idx_end = 2000

plt.figure()
plt.plot(time[idx_start:idx_end], upp_leg_angle_acc_x[idx_start:idx_end], label="Hoek uit Acc X")
plt.title("Bovenbeenhoek op basis van Acc X (sample 200-2000)")
plt.xlabel("Tijd (s)")
plt.ylabel("Hoek t.o.v. verticaal (graden)")
plt.legend()
plt.grid(True)


# %% Opdracht 2e (0.5 p) 
# Low-pass filter het versnellingssignaal van X met een 4e order Butterworth filter en een Cut-off frequentie van 5 Hz. 
# Sla dit gefilterde signaal op als “acc_x_lp”. En bereken nu de hoek (in graden) van de sensor ten opzichte van de verticaal, op basis van dit gefilterde signaal. 
# Komt er nu ook nog een waarschuwing bij het runnen van de code? 
# Plot deze nieuw berekende hoek in dezelfde plot als 2d (ook van sample 200 – 2000).

# %% Opdracht 2f (0.5 p) 
# Bereken de hoekversnelling op basis van Gyro Z signaal, en sla deze op als “angular_acc_gyro_z”. 


# %% Opdracht 2g (0.5 p)  
# Geef de code voor het vinden van de hoogste hoekversnelling (angular_acc_gyro_z), zowel de waarde, als het tijdstip.  

# %% Opdracht 2h (0.5 p) 
# Plot “angular_acc_gyro_z” tegen de tijd en plot een rode stip/rondje rond de gevonden piek. Voeg as-labels en een legenda toe.