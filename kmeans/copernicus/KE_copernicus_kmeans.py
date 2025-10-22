import os
os.environ['OMP_NUM_THREADS'] = '13'

import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cmocean
from kmeans_ci import kmeans_ci_py
from scipy.io import loadmat

# Parámetros temporales
seasons = 'Jan'
midmon = 'Jun'
seasone = 'Dec'
yeari = 1993
yeare = 2023

slonmin, slonmax = -84, -70
slatmin, slatmax = 7, 16
pdlatmin, pdlatmax = slatmin, slatmax
pdlonmin, pdlonmax = slonmin, slonmax

# Parámetros de clúster
minclust, maxclust = 2, 10
varfract, nclust, nsim = 0.95, 10, 100

# Directorio de trabajo
os.chdir(r'C:\LAURA\NUEVOS_DATASE_2018-2023\filtrado_Caribe')

# Carga de datos espaciales
lonu = sio.loadmat(r'C:\LAURA\Patronesclimaticos\lonur.mat')['vlon'].flatten()
latu = sio.loadmat(r'C:\LAURA\Patronesclimaticos\latur.mat')['vlat'].flatten()

up = loadmat("u_caribe_actual.mat")["u"]
vp = loadmat("v_caribe_actual.mat")["v"]

# Fechas de inicio y fin del dataset
start_date_dataset = np.datetime64('2000-01-01')
start_date = np.datetime64('2000-01-01')
end_date = np.datetime64('2023-12-31')

days_from_start_to_begin = (start_date - start_date_dataset).astype(int)
days_from_start_to_end = (end_date - start_date_dataset).astype(int)
mask = np.arange(days_from_start_to_begin, days_from_start_to_end + 1)

# Filtrar datos
up, vp = up[:, mask], vp[:, mask]

up1, vp1, up2, vp2 = up.copy(), vp.copy(), up.copy(), vp.copy()

# Limpieza de NaN
mask_nan = np.isnan(up) | np.isnan(vp)
TF = np.isnan(up[:, 0])
indf = np.where(TF)[0]
up, vp = np.delete(up, indf, axis=0), np.delete(vp, indf, axis=0)
TF1 = np.isnan(vp[:, 0])
indf1 = np.where(TF1)[0]
vp, up = np.delete(vp, indf1, axis=0), np.delete(up, indf1, axis=0)

# Energía Cinética (KE)
KE = np.full_like(up, np.nan)
for t in range(up.shape[1]):
    try:
        u, v = up[:, t], vp[:, t]
        if np.isnan(u).any() or np.isnan(v).any():
            raise ValueError(f"Valores NaN detectados en el día {t}")
        KE[:, t] = 0.5 * (u**2 + v**2)
    except Exception as e:
        print(f"Error en el cálculo de energía cinética para el día {t}: {e}")

# Restaurar forma original de KE
KE_restored = np.full(mask_nan.shape, np.nan)
KE_restored[~mask_nan] = KE.flatten()

print('Circulation variable (currents) has been read and storaged.')

# K-means sobre KE
uvn = KE.T
K = np.zeros((8766, maxclust))
CI = np.full((maxclust, 1), np.nan)
np.random.seed(1)

for kk in range(minclust, maxclust + 1):
    np.random.seed(1)
    CI[kk - 1], K[:, kk - 1] = kmeans_ci_py(uvn, 's', varfract, kk, 100)
    print('Se hizo el cluster')

# Cálculo de clústeres de KE
nclust = 9
indx = K[:, nclust - 1]
KE_clust_reshaped = np.zeros((nclust, 109, 169))

for k in range(nclust):
    kk = np.where(indx == k)[0]
    KE_clust_reshaped[k, :, :] = np.nanmean(
        KE_restored[:, kk], axis=1
    ).reshape((169, 109), order='F').T

vorticity_max = np.nanmax(KE_clust_reshaped)
vorticity_min = np.nanmin(KE_clust_reshaped)
print(f"Máximo EKE: {vorticity_max}")
print(f"Mínimo EKE: {vorticity_min}")

# Frecuencia y probabilidad de clústeres
fechas = pd.date_range(start='2000-01-01', end='2023-12-31')
meses = fechas.month
frecuencia_clusters = np.zeros((nclust, 12))

for k in range(nclust):
    dias_cluster_k = meses[indx == k]
    for mes in range(1, 13):
        frecuencia_clusters[k, mes - 1] = np.sum(dias_cluster_k == mes)

eventos_por_cluster = np.sum(frecuencia_clusters, axis=1)
probabilidad_clusters = (eventos_por_cluster / np.sum(eventos_por_cluster)) * 100

# FIGURA 1: Mapas de Energía Cinética
vmin, vmax = 0, 0.2
ncols = min(3, nclust)
nrows = int(np.ceil(nclust / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5), constrained_layout=True)
axes = axes.flatten()

for kplot in range(nclust):
    KE_map = KE_clust_reshaped[kplot, :, :]
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

# FIGURA 2: Heatmap de frecuencia mensual
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
