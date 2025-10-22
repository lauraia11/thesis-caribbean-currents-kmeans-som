from scipy.io import loadmat, savemat
import numpy as np
import pandas as pd

# Cargar los datos de latitud y longitud (OSCAR)
data = loadmat(r'C:/LAURA/oscar_combined.mat')  # Usa una ruta raw string para evitar problemas con "\"
lat = data['lat']
lon = data['lon']

# Crear las matrices de grilla (LO, LA)
lon, lat = np.meshgrid(lon, lat)

# Cargar los datos de latitud y longitud (HYCOM)
file = 'C:/LAURA/vorticidadkmeans/HYCOM/latitud_hycom2d.mat'
file2 = 'C:/LAURA/vorticidadkmeans/HYCOM/longitud_hycom2d.mat'
file_data = loadmat(file)
file_data2 = loadmat(file2)

lat = file_data['latitude_2d']
lon = file_data2['longitude_2d']

def calculate_pm_pn_with_borders(lat, lon):
    # Radio de la Tierra en metros
    R = 6371e3  # 6371 km en metros
    
    # Convertir grados a radianes
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Calcular las diferencias de latitud y longitud
    dlat = np.diff(lat_rad, axis=0)  # Diferencias en la dirección de latitud
    dlon = np.diff(lon_rad, axis=1)  # Diferencias en la dirección de longitud
    
    # Calcular los promedios de latitudes
    lat_mean = (lat_rad[:-1, :-1] + lat_rad[1:, :-1]) / 2  # Promedio en dirección latitud
    
    # Ajustar dimensiones para dlon y dlat
    dlat_adjusted = dlat[:, :-1]  # Recortar para coincidir con lat_mean
    dlon_adjusted = dlon[:-1, :]  # Recortar para coincidir con lat_mean
    
    # Calcular las distancias en metros
    dx = R * np.cos(lat_mean) * dlon_adjusted  # Distancia en dirección longitudinal
    dy = R * dlat_adjusted  # Distancia en dirección latitudinal
    
    # Calcular pm y pn
    pm = 1 / dx  # Inverso de la distancia en x
    pn = 1 / dy  # Inverso de la distancia en y
    
    # Manejo de bordes para mantener dimensiones
    # Añadir fila y columna al final para igualar dimensiones originales
    pm_with_borders = np.zeros_like(lat)
    pn_with_borders = np.zeros_like(lat)
    
    pm_with_borders[:-1, :-1] = pm
    pm_with_borders[-1, :-1] = pm[-1, :]  # Copiar última fila
    pm_with_borders[:, -1] = pm_with_borders[:, -2]  # Copiar última columna
    
    pn_with_borders[:-1, :-1] = pn
    pn_with_borders[-1, :-1] = pn[-1, :]  # Copiar última fila
    pn_with_borders[:, -1] = pn_with_borders[:, -2]  # Copiar última columna
    
    return pm_with_borders, pn_with_borders

# Calcular pm y pn con manejo de bordes
pm, pn = calculate_pm_pn_with_borders(lat, lon)

# Aplanar pm y pn
pm_flat = pm.ravel(order='C')
pn_flat = pn.ravel(order='C')

data = loadmat('C:/LAURA/Laura_isabela/Hycom_Diarios_Caribe_1993_2019.mat')
tiempo = data['tiempo']
meses = tiempo[:,2]
# Calcular la dimensión temporal (número de días)
#fechas = pd.date_range(start='2000/01/01', end='2023/12/31')  # Rango de fechas
ntime = len(meses)  # Número de días

# Replicar para la dimensión temporal
pm_replicated = np.tile(pm_flat, (ntime, 1)).T  # Expandir a (nspatial, ntime)
pn_replicated = np.tile(pn_flat, (ntime, 1)).T  # Expandir a (nspatial, ntime)

import h5py

# Guardar los arrays en un archivo .mat usando h5py
with h5py.File('C:/LAURA/kemans_23anos/OSCAR/pnpm/pm_OSCAR_2d.mat', 'w') as file:
    file.create_dataset('pm', data=pm_replicated)
    
with h5py.File('C:/LAURA/kemans_23anos/OSCAR/pnpm/pn_OSCAR_2d.mat', 'w') as file:
    file.create_dataset('pn', data=pn_replicated)

print(np.isnan(pn).any(), np.isnan(lon).any())  # Verifica si hay valores NaN
print(lat.min(), lat.max(), lon.min(), lon.max())  # Revisa rangos válidos

