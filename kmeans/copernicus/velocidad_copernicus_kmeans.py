import os
os.environ['OMP_NUM_THREADS'] = '13'

import numpy as np
import h5py
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from kmeans_ci import kmeans_ci_py
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Parámetros
slonmin, slonmax = -84, -70
slatmin, slatmax = 7, 16
minclust, maxclust = 2, 10
varfract, nsim = 0.95, 100

# Directorio y carga de datos
os.chdir(r'C:\LAURA\NUEVOS_DATASE_2018-2023\filtrado_Caribe')
lonu = sio.loadmat(r'C:\LAURA\Patronesclimaticos\lonur.mat')['vlon'].flatten()
latu = sio.loadmat(r'C:\LAURA\Patronesclimaticos\latur.mat')['vlat'].flatten()

up = sio.loadmat('u_caribe_actual.mat')["u"]
vp = sio.loadmat('v_caribe_actual.mat')["v"]

# Enmascarado temporal 2000–2023
start_date_dataset = np.datetime64('1993-01-01')
mask = np.arange((np.datetime64('2000-01-01') - start_date_dataset).astype(int),
                 (np.datetime64('2023-12-31') - start_date_dataset).astype(int) + 1)
up, vp = up[:, mask], vp[:, mask]

# Eliminar filas con NaN
TF = np.isnan(up[:, 0])
up, vp = np.delete(up, TF, axis=0), np.delete(vp, TF, axis=0)
TF1 = np.isnan(vp[:, 0])
vp, up = np.delete(vp, TF1, axis=0), np.delete(up, TF1, axis=0)

# Construcción de muestras y cálculo de clusters
uv = np.concatenate((up.T, vp.T), axis=1)
K = np.zeros((8766, maxclust))
CI = np.full((maxclust, 1), np.nan)
np.random.seed(1)

for kk in range(minclust, maxclust + 1):
    np.random.seed(1)
    CI[kk - 1], K[:, kk - 1] = kmeans_ci_py(uv, 's', varfract, kk, 100)
    print('Se hizo el cluster')

nclust = 9
indx = K[:, nclust - 1]
uqcompa = np.zeros((nclust, up.shape[0]))
vqcompa = np.zeros((nclust, vp.shape[0]))

for k in range(nclust):
    kk = np.where(indx == k)[0]
    uqcompa[k, :] = np.nanmean(up[:, kk], axis=1)
    vqcompa[k, :] = np.nanmean(vp[:, kk], axis=1)

uqcompa2 = np.reshape(uqcompa, (nclust, 169, 109), order='F')
vqcompa2 = np.reshape(vqcompa, (nclust, 169, 109), order='F')
uqcompa2 = np.flip(np.rot90(uqcompa2, k=-1, axes=(1, 2)), axis=2)
vqcompa2 = np.flip(np.rot90(vqcompa2, k=-1, axes=(1, 2)), axis=2)
vemap_list = []
for k in range(nclust):
    umap = np.squeeze(uqcompa2[k, :, :])
    vmap = np.squeeze(vqcompa2[k, :, :])
    vemap_list.append(np.sqrt(umap**2 + vmap**2))

# Probabilidades y frecuencia mensual
fechas = pd.date_range(start='2000-01-01', end='2023-12-31')
meses = fechas.month
eventos_totales = len(indx)
eventos_por_cluster = np.array([np.sum(indx == k) for k in range(nclust)])
probabilidad_clusters = (eventos_por_cluster / eventos_totales) * 100

frecuencia_clusters = np.zeros((nclust, 12))
for k in range(nclust):
    dias_k = meses[indx == k]
    for m in range(1, 13):
        frecuencia_clusters[k, m - 1] = np.sum(dias_k == m)

# FIGURA 1: Mapas 3×3 con streamlines
vmin, vmax = 0, 0.9
colors = [
    (0.1, 0.2, 0.7),
    (0.3, 0.7, 1.0),
    (1.0, 1.0, 0.4),
    (1.0, 0.5, 0.0),
    (0.8, 0.0, 0.0)
]
custom_cmap = LinearSegmentedColormap.from_list("CustomPalette", colors, N=256)
ncols = 3
nrows = int(np.ceil(nclust / ncols)) 

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 6),
                         subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes.flatten()
land = cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='face', facecolor='lightgrey')

for k in range(nclust):
    vemap = vemap_list[k]
    Xvec = np.linspace(slonmin, slonmax, vemap.shape[1])
    Yvec = np.linspace(slatmin, slatmax, vemap.shape[0])
    Xg, Yg = np.meshgrid(Xvec, Yvec)
    ax = axes[k]
    ax.add_feature(land)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    cont = ax.contourf(Xg, Yg, vemap, levels=np.linspace(vmin, vmax, 1024),
                       cmap=custom_cmap, extend='both', transform=ccrs.PlateCarree())
    ax.streamplot(Xg, Yg, np.squeeze(uqcompa2[k]), np.squeeze(vqcompa2[k]),
                  color='black', linewidth=0.8, density=3.0, transform=ccrs.PlateCarree())
    ax.set_xlim([slonmin, slonmax])
    ax.set_ylim([slatmin, slatmax])
    if k % ncols == 0:
        ax.set_yticks(range(slatmin, slatmax + 1, 2))
        ax.set_yticklabels([f'{lat}°N' for lat in range(slatmin, slatmax + 1, 2)])
    else:
        ax.set_yticks([])
    if k >= (nrows - 1) * ncols:
        ax.set_xticks(range(slonmin, slonmax + 1, 2))
        ax.set_xticklabels([f'{lon}°W' for lon in range(slonmin, slonmax + 1, 2)])
    else:
        ax.set_xticks([])
    ax.text(slonmax - 0.5, slatmin + 0.5, f'Clúster {k + 1}\n {probabilidad_clusters[k]:.2f}%',
            fontsize=12, color='black', ha='right', va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

fig.suptitle('Velocidad Kmeans - COPERNICUS', fontsize=20, fontweight='bold', style='italic', family='serif', y=0.96)
cbar = fig.colorbar(cont, ax=axes, orientation='vertical', fraction=0.02, pad=0.05)
cbar.set_label('Velocidad [m/s]')
plt.tight_layout(rect=[0.1, 0.1, 0.85, 0.95])
plt.show()

# FIGURA 2: Heatmap mensual
plt.figure(figsize=(8, 5))
ax = sns.heatmap(frecuencia_clusters, cmap='Greys', annot=False, cbar_kws={'label': 'Frecuencia'}, linewidths=0.5)
ax.set_xlabel('Mes')
ax.set_ylabel('Clúster')
ax.set_xticklabels([str(i) for i in range(1, 13)])
ax.set_yticklabels([f'{i + 1}' for i in range(nclust)])
plt.title('Frecuencia de Clúster por Mes', fontsize=16)
plt.tight_layout()
plt.show()
