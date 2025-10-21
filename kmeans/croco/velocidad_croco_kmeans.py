import os
os.environ['OMP_NUM_THREADS'] = '30'

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
slonmin, slonmax = -84, -71
slatmin, slatmax = 7, 15.0098
minclust, maxclust = 2, 10
varfract, nsim = 0.95, 100

# Datos
os.chdir(r'C:\LAURA\CROCOFINAL')
lonu = sio.loadmat(r'C:\LAURA\CROCOFINAL\longitud_croco.mat')['longitud_transpuesta'].flatten()
latu = sio.loadmat(r'C:\LAURA\CROCO\latitud_reorganizada.mat')['latitud_organizada'].flatten()

with h5py.File(r'C:/LAURA/CROCOFINALFINAL/arrays_u_final1.mat', 'r') as f:
    up = f['array_upt'][:]
with h5py.File(r'C:/LAURA/CROCOFINALFINAL/arrays_v_final1.mat', 'r') as f:
    vp = f['array_vpt'][:]

# Copias (espacio x tiempo)
up1, vp1 = up.copy(), vp.copy()
up2, vp2 = up.copy(), vp.copy()

# Eliminar filas con NaN en cualquier columna
up = up[~np.isnan(up).any(axis=1)]
vp = vp[~np.isnan(vp).any(axis=1)]
up1, vp1 = up.copy(), vp.copy()

# Matriz de muestras (tiempo, 2*espacio) y K para 2000-01-01..2023-12-31 (8766 días)
uv = np.concatenate((up1.T, vp1.T), axis=1)
uvn = uv.copy()
K = np.zeros((8766, maxclust))
CI = np.full((maxclust, 1), np.nan)
stan='s'
prop=varfract
nclus=10
np.random.seed(1)

print('Circulation variable (currents) has been read and storaged.')
for kk in range(minclust, maxclust + 1):
   
   np.random.seed(1)
   CI[kk - 1], K[:, kk - 1] = kmeans_ci_py(uvn, 's', varfract, kk, 100)
   print('Se hizo el cluster')

# Selección k=9
nclust = 9
indx = K[:, nclust - 1]

# Compuestos U/V por clúster
nday = np.zeros(nclust)
uqcompa = np.zeros((nclust, up2.shape[0]))
vqcompa = np.zeros((nclust, vp2.shape[0]))

for k in range(nclust):
    kk = np.where(indx == k)[0]
    nday[k] = len(kk)
    uqcompa[k, :] = np.nanmean(up2[:, kk], axis=1)
    vqcompa[k, :] = np.nanmean(vp2[:, kk], axis=1)

# Rejillas y magnitud de velocidad
uqcompa2 = np.reshape(uqcompa, (nclust, 295, 469), order='F').transpose(0, 1, 2)
vqcompa2 = np.reshape(vqcompa, (nclust, 295, 469), order='F').transpose(0, 1, 2)
vemap_list = []
for k in range(nclust):
    umap = np.squeeze(uqcompa2[k, :, :])
    vmap = np.squeeze(vqcompa2[k, :, :])
    vemap_list.append(np.sqrt(umap**2 + vmap**2))

Xmat = np.reshape(lonu, (295, 469)).T
Ymat = np.reshape(latu, (295, 469)).T

# Probabilidades y frecuencias mensuales
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

# FIGURA 1: Mapas 3x3 con streamlines
vmin, vmax = 0, 0.9
colors = [
    (0.1, 0.2, 0.7),	# Azul profundo para valores bajos
    (0.3, 0.7, 1.0),	# Azul claro
    (1.0, 1.0, 0.4),	# Amarillo para valores intermedios
    (1.0, 0.5, 0.0),	# Naranja
    (0.8, 0.0, 0.0)	# Rojo intenso para valores altos
]
custom_cmap = LinearSegmentedColormap.from_list("CustomPalette", colors, N=256)

ncols = 3
nrows = (nclust // 3) + (nclust % 3 > 0)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5),
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
    ax.streamplot(Xg, Yg, np.squeeze(uqcompa2[k, :, :]), np.squeeze(vqcompa2[k, :, :]),
                  color='black', linewidth=0.8, density=3.0, transform=ccrs.PlateCarree())

    ax.set_xlim([slonmin, slonmax])
    ax.set_ylim([slatmin, slatmax])

    if k % ncols == 0:
        ax.set_yticks(np.arange(slatmin, slatmax + 1, 2))
        ax.set_yticklabels([f'{lat}°N' for lat in np.arange(slatmin, slatmax + 1, 2)])
    else:
        ax.set_yticks([]); ax.set_yticklabels([])

    if k >= (nrows - 1) * ncols:
        ax.set_xticks(np.arange(slonmin, slonmax + 1, 2))
        ax.set_xticklabels([f'{lon}°W' for lon in np.arange(slonmin, slonmax + 1, 2)])
    else:
        ax.set_xticks([]); ax.set_xticklabels([])

    ax.text(slonmax - 0.5, slatmin + 0.5,
            f'Clúster {k + 1}\n {probabilidad_clusters[k]:.2f}%',
            fontsize=12, color='black', ha='right', va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

fig.suptitle('Velocidad Kmeans - CROCO', fontsize=20, fontweight='bold', style='italic', family='serif', y=0.96)
cbar = fig.colorbar(cont, ax=axes, orientation='vertical', fraction=0.02, pad=0.05)
cbar.set_label('Velocidad [m/s]')
plt.tight_layout(rect=[0.1, 0.1, 0.85, 0.95])
plt.show()

# FIGURA 2: Heatmap de frecuencia mensual
plt.figure(figsize=(8, 5))
ax = sns.heatmap(frecuencia_clusters, cmap='Greys', annot=False, cbar_kws={'label': 'Frecuencia'}, linewidths=0.5)
ax.set_xlabel('Mes')
ax.set_ylabel('Clúster')
ax.set_xticklabels([str(i) for i in range(1, 13)])
ax.set_yticklabels([f'{i + 1}' for i in range(nclust)])
plt.title('Frecuencia de Clúster por Mes', fontsize=16)
plt.tight_layout()
plt.show()
