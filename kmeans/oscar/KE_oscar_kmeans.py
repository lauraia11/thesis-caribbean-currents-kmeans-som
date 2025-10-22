# ENERGÍA CINÉTICA PARA OSCAR
import os
os.environ['OMP_NUM_THREADS'] = '6'

import numpy as np
import h5py
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cmocean
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from kmeans_ci import kmeans_ci_py
from scipy.io import loadmat

# Parámetros temporales
seasons, midmon, seasone = 'Jan', 'Jun', 'Dec'
slonmin, slonmax, slatmin, slatmax = -84, -72, 8.33333, 18.3333
pdlatmin, pdlatmax = slatmin, slatmax
pdlonmin, pdlonmax = slonmin, slonmax

# Parámetros de clúster
minclust, maxclust = 2, 10
varfract, nclust, nsim = 0.95, 10, 100

# Carga de datos
data = loadmat(r'C:/LAURA/oscar_combined.mat')
lat1, lon1 = data['lat'], data['lon']
Lon, Lat = np.meshgrid(lon1, lat1)
tiempo = data['tiempo']

up = sio.loadmat(r'C:\LAURA\OSCAR1cuarto\U_oscar2final.mat')['u']
vp = sio.loadmat(r'C:\LAURA\OSCAR1cuarto\V_oscar2final.mat')['v']

# Limpieza de NaN
valid_rows = ~np.isnan(up).any(axis=1)
mask_nan = ~valid_rows
up, vp = up[valid_rows], vp[valid_rows]

print("NaN en UP:", np.isnan(up).any())
print("NaN en VP:", np.isnan(vp).any())

# Energía Cinética (KE)
nspatial, ntime = up.shape
KE = np.full_like(up, np.nan)
for t in range(ntime):
    try:
        u, v = up[:, t], vp[:, t]
        if np.isnan(u).any() or np.isnan(v).any():
            raise ValueError(f"Valores NaN detectados en el día {t}")
        KE[:, t] = 0.5 * (u**2 + v**2)
    except Exception as e:
        print(f"Error en el cálculo de energía cinética para el día {t}: {e}")

KE_restored = np.full((mask_nan.shape[0], ntime), np.nan)
KE_restored[~mask_nan, :] = KE

# Aplicación de K-means
uvn = KE.T
K = np.zeros((8766, maxclust))
CI = np.full((maxclust, 1), np.nan)
np.random.seed(1)

for kk in range(minclust, maxclust + 1):
    np.random.seed(1)
    CI[kk - 1], K[:, kk - 1] = kmeans_ci_py(uvn, 's', varfract, kk, 100)
    print('Se hizo el cluster')

# Cálculo de clústeres KE
nclust = 9
indx = K[:, nclust - 1]
KE_clust_reshaped = np.zeros((nclust, 53, 33))

for k in range(nclust):
    kk = np.where(indx == k)[0]
    KE_clust_reshaped[k, :, :] = np.nanmean(
        KE_restored[:, kk], axis=1).reshape((33, 53), order='C').T

KE_clust_reshaped = np.rot90(KE_clust_reshaped, k=-2, axes=(1, 2))
KE_clust_reshaped = np.flip(KE_clust_reshaped, axis=1)

KE_max, KE_min = np.nanmax(KE_clust_reshaped), np.nanmin(KE_clust_reshaped)
print(f"Máximo EKE: {KE_max}")
print(f"Mínimo EKE: {KE_min}")

# Frecuencias y probabilidades
meses = tiempo[:, 1]
frecuencia_clusters = np.zeros((nclust, 12))

for k in range(nclust):
    dias_cluster_k = meses[indx == k]
    for mes in range(1, 13):
        frecuencia_clusters[k, mes - 1] = np.sum(dias_cluster_k == mes)

eventos_por_cluster = np.sum(frecuencia_clusters, axis=1)
probabilidad_clusters = (eventos_por_cluster / np.sum(eventos_por_cluster)) * 100

# FIGURA 1: Mapas de Energía Cinética (EKE)
vmin, vmax = 0, 0.2
ncols = min(3, nclust)
nrows = int(np.ceil(nclust / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5), constrained_layout=True)
axes = axes.flatten()

for kplot in range(nclust):
    KE_map = np.rot90(KE_clust_reshaped[kplot, :, :], k=1)
    probabilidad = probabilidad_clusters[kplot]
    Xvec = np.linspace(slonmin, slonmax, KE_map.shape[1])
    Yvec = np.linspace(slatmin, slatmax, KE_map.shape[0])
    Xmat, Ymat = np.meshgrid(Xvec, Yvec)
    levels = np.linspace(vmin, vmax, 100)

    contf = axes[kplot].contourf(
        Xmat, Ymat, KE_map,
        levels=levels,
        cmap=cmocean.cm.balance,
        extend='both'
    )
    axes[kplot].set_xlim([slonmin, slonmax])
    axes[kplot].set_ylim([slatmin, slatmax])
    axes[kplot].set_title(f"Cluster {kplot + 1} ({probabilidad:.2f}%)", fontsize=10)

cbar = fig.colorbar(
    contf,
    ax=axes,
    orientation='horizontal',
    fraction=0.05,
    pad=0.1
)
cbar.set_label(r'Energía Cinética (EKE) [m²/s²] ×10⁻²', fontsize=12)
cbar.set_ticks(np.linspace(vmin, vmax, 5))
cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x * 1e2:.1f}'))
plt.show()

# FIGURA 2: Heatmap mensual
plt.figure(figsize=(8, 5))
ax = sns.heatmap(
    frecuencia_clusters, cmap='Greys', annot=False,
    cbar_kws={'label': 'Frecuencia'}, linewidths=0.5
)
ax.set_xlabel('Mes')
ax.set_ylabel('Clúster')
ax.set_xticklabels([str(i) for i in range(1, 13)])
ax.set_yticklabels([f'{i + 1}' for i in range(nclust)])
plt.title('Frecuencia de Clúster por Mes', fontsize=16)
plt.tight_layout()
plt.show()
