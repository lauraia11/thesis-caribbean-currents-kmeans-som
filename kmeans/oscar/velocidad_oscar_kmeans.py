import os
os.environ['OMP_NUM_THREADS'] = '6'

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
slatmin, slatmax = 7, 15
minclust, maxclust = 2, 10
varfract, nsim = 0.95, 100

# Carga de datos
data = sio.loadmat(r'C:/LAURA/oscar_combined.mat')
lat, lon = data['lat'], data['lon']
LO, LA = np.meshgrid(lon, lat)

u_data = sio.loadmat(r'C:\LAURA\OSCAR1cuarto\U_oscar2final.mat')
v_data = sio.loadmat(r'C:\LAURA\OSCAR1cuarto\V_oscar2final.mat')
up, vp = u_data['u'], v_data['v']

# Limpieza de NaN
TF = np.isnan(up[:, 0])
up, vp = np.delete(up, TF, axis=0), np.delete(vp, TF, axis=0)
TF1 = np.isnan(vp[:, 0])
vp, up = np.delete(vp, TF1, axis=0), np.delete(up, TF1, axis=0)
up, vp = up[~np.isnan(up).any(axis=1)], vp[~np.isnan(vp).any(axis=1)]

# Matriz de muestras (tiempo, 2*espacio)
uv = np.concatenate((up.T, vp.T), axis=1)
K = np.zeros((8766, maxclust))
CI = np.full((maxclust, 1), np.nan)
np.random.seed(1)

for kk in range(minclust, maxclust + 1):
    np.random.seed(1)
    CI[kk - 1], K[:, kk - 1] = kmeans_ci_py(uv, 's', varfract, kk, 100)
    print(f'Clúster {kk} completado')

# Cálculo de compuestos U/V
nclust = 9
indx = K[:, nclust - 1]
uqcompa = np.zeros((nclust, up.shape[0]))
vqcompa = np.zeros((nclust, vp.shape[0]))

for k in range(nclust):
    kk = np.where(indx == k)[0]
    uqcompa[k, :] = np.nanmean(up[:, kk], axis=1)
    vqcompa[k, :] = np.nanmean(vp[:, kk], axis=1)

uqcompa2 = np.reshape(uqcompa, (nclust, 33, 53), order='C').transpose(0, 1, 2)
vqcompa2 = np.reshape(vqcompa, (nclust, 33, 53), order='C').transpose(0, 1, 2)

vemap_list = [None]*nclust
for kplot in range(nclust):
    umap = np.squeeze(uqcompa2[kplot, :, :])
    vmap = np.squeeze(vqcompa2[kplot, :,:])
    
    vemap_list[kplot] = np.sqrt(umap**2 + vmap**2)   #Cálculo de la magnitud de la velocidad raiz(X^2+Y^2)
umap=umap.T
vmap=vmap.T

# Probabilidad y frecuencia mensual
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

# FIGURA 1: Mapas de velocidad y streamlines
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
    ax.streamplot(Xg, Yg, np.squeeze(uqcompa2[k]), np.squeeze(vqcompa2[k]),
                  color='black', linewidth=0.8, density=3.0, transform=ccrs.PlateCarree())
    ax.set_xlim([slonmin, slonmax])
    ax.set_ylim([slatmin, slatmax])
    if k % ncols == 0:
        ax.set_yticks(np.arange(slatmin, slatmax + 1, 2))
        ax.set_yticklabels([f'{lat}°N' for lat in np.arange(slatmin, slatmax + 1, 2)])
    else:
        ax.set_yticks([])
    if k >= (nrows - 1) * ncols:
        ax.set_xticks(np.arange(slonmin, slonmax + 1, 2))
        ax.set_xticklabels([f'{lon}°W' for lon in np.arange(slonmin, slonmax + 1, 2)])
    else:
        ax.set_xticks([])
    ax.text(slonmax - 0.5, slatmin + 0.5, f'Clúster {k + 1}\n {probabilidad_clusters[k]:.2f}%',
            fontsize=12, color='black', ha='right', va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

fig.suptitle('Velocidad Kmeans - OSCAR', fontsize=20, fontweight='bold', style='italic', family='serif', y=0.9)
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
