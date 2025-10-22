import os
os.environ['OMP_NUM_THREADS'] = '13'

import numpy as np
import h5py
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.io as sio
from kmeans_ci import kmeans_ci_py
import cmocean
from scipy.io import savemat
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors

#Define parámetros
seasons = 'Jan'
midmon = 'Jun'
seasone = 'Dec'
yeari = 2000
yeare = 2023

slonmin = -84
slonmax = -71
slatmin = 7
slatmax = 15.0098

pdlatmin = slatmin
pdlatmax = slatmax
pdlonmin = slonmin
pdlonmax = slonmax

#Parámetros de cluster
minclust = 2
maxclust = 10
varfract = 0.95
nclust = 10
nsim = 100

os.chdir(r'C:\LAURA\CROCOFINAL')

lonu_data = sio.loadmat(r'C:\LAURA\CROCOFINAL\longitud_croco.mat')
latu_data = sio.loadmat(r'C:\LAURA\CROCO\latitud_reorganizada.mat')

lonu = lonu_data['longitud_transpuesta'].flatten()
latu = latu_data['latitud_organizada'].flatten()
nlon2 = len(lonu)
nlat2 = len(latu)

with h5py.File('C:/LAURA/vorticidadkmeans/Croco/pn_croco_2d.mat', 'r') as file:
    pn = file['pn'][:]

with h5py.File('C:/LAURA/vorticidadkmeans/Croco/pm_croco_2d.mat', 'r') as file:
    pm = file['pm'][:]

with h5py.File('C:/LAURA/CROCOFINALFINAL/arrays_u_final1.mat', 'r') as file:
    up = file['array_upt'][:]

with h5py.File('C:/LAURA/CROCOFINALFINAL/arrays_v_final1.mat', 'r') as file:
    vp = file['array_vpt'][:]

up1 = up.copy()
vp1 = vp.copy()
up2 = up.copy()
vp2 = vp.copy()

pn1 = pn.copy()
pm1 = pm.copy()

# Crear una máscara de NaN en los datos originales de u y v
mask_nan = np.isnan(up) | np.isnan(vp)

TF = np.isnan(up[:, 0])
indf = np.where(TF)[0]
up = np.delete(up, indf, axis=0)
vp = np.delete(vp, indf, axis=0)
pn = np.delete(pn, indf, axis=0)
pm = np.delete(pm, indf, axis=0)

TF1 = np.isnan(vp[:, 0])
indf1 = np.where(TF1)[0]
vp = np.delete(vp, indf1, axis=0)
up = np.delete(up, indf1, axis=0)

os.chdir(r'C:\LAURA')
line2000_shore = np.loadtxt('line2000_shore')

vp1 = vp.copy()
up1 = up.copy()

# Dimensiones espaciales y temporales
nspatial = up.shape[0]
ntime = up.shape[1]

vorticity = np.full_like(up, np.nan)

for t in range(ntime):
    try:
        # Extraer u, v, pm, pn para el instante t
        u = up[:, t]
        v = vp[:, t]
        pm_t = pm[:, t]
        pn_t = pn[:, t]

        if np.isnan(u).any() or np.isnan(v).any():
            raise ValueError(f"Valores NaN detectados en el día {t}")

        # Calcular derivadas espaciales con pm y pn
        dV_dx = pm_t * (np.roll(v, shift=-1) - v)
        dU_dy = pn_t * (np.roll(u, shift=-1) - u)

        # Calcular vorticidad
        vorticity[:, t] = dV_dx - dU_dy
    except Exception as e:
        print(f"Error en el cálculo de vorticidad para el día {t}: {e}")

# Restaurar la forma original de la vorticidad
vorticity_restored = np.full(mask_nan.shape, np.nan)
vorticity_restored[~mask_nan] = vorticity.flatten()

print('Circulation variable (currents) has been read and storaged.')

# el tiempo es 8766 el tiempo siempre debe ir a la izquierda en uvn y se debe usar para definir el tamaño de K
uv = vorticity.copy()
uvn = uv.T

if np.isnan(vorticity).any():
    print("uvn contiene valores NaN.")

nan_positions = np.argwhere(np.isnan(uvn))
print(f"Posiciones de NaN: {nan_positions}")

inf_positions = np.argwhere(np.isinf(uvn))
print(f"Posiciones de Inf: {inf_positions}")

K = np.zeros((8766, maxclust))
X = uvn
CI = np.full((maxclust, 1), np.nan)

stan = 's'
prop = varfract
nclus = nclust
np.random.seed(1)

for kk in range(minclust, maxclust + 1):
    np.random.seed(1)
    CI[kk - 1], K[:, kk - 1] = kmeans_ci_py(uvn, 's', varfract, kk, 100)
    print('Se hizo el cluster')

# Preparar matrices para los resultados de clustering
nclust = 9
indx = K[:, nclust - 1]

vorticity_clust_reshaped = np.zeros((nclust, 295, 469))

for k in range(nclust):
    kk = np.where(indx == k)[0]
    vorticity_clust_reshaped[k, :, :] = np.nanmean(
        vorticity_restored[:, kk], axis=1
    ).reshape((469, 295), order='C').T

vorticity_max = np.nanmax(vorticity_clust_reshaped)
vorticity_min = np.nanmin(vorticity_clust_reshaped)

print(f"El valor más alto de vorticidad es: {vorticity_max}")
print(f"El valor más bajo de vorticidad es: {vorticity_min}")

#----------MAPA-------------------------------
vmin, vmax = -0.00001643, 0.00001643
cmap = cmocean.cm.curl
labs = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']

Xvec = np.linspace(slonmin, slonmax, vorticity_clust_reshaped.shape[2])
Yvec = np.linspace(slatmin, slatmax, vorticity_clust_reshaped.shape[1])
Xmat, Ymat = np.meshgrid(Xvec, Yvec)

# Frecuencias y probabilidades del clúster
fechas = pd.date_range(start='2000/01/01', end='2023/12/31')
meses = fechas.month

indx = K[:, nclust - 1]
frecuencia_clusters = np.zeros((nclust, 12))

for k in range(nclust):
    dias_cluster_k = meses[indx == k]
    for mes in range(1, 13):
        frecuencia_clusters[k, mes - 1] = np.sum(dias_cluster_k == mes)

eventos_por_cluster = np.sum(frecuencia_clusters, axis=1)
eventos_totales = np.sum(eventos_por_cluster)
probabilidad_clusters = (eventos_por_cluster / eventos_totales) * 100

#--------------------------------------------------
class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_value = super().__call__(value, clip)
        result = np.ma.masked_array(
            (normalized_value - 0.5) * (self.vmax - self.vmin)
            / (self.vmax - self.midpoint) + 0.5
        )
        return result

ncols = min(3, nclust)
nrows = int(np.ceil(nclust / ncols))
norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0.0)

fig, axes = plt.subplots(
    nrows, ncols, figsize=(ncols * 6, nrows * 5), constrained_layout=True
)
axes = axes.flatten()

for kplot in range(nclust):
    vorticity_map = vorticity_clust_reshaped[kplot, :, :]
    probabilidad = probabilidad_clusters[kplot]
    Xvec = np.linspace(slonmin, slonmax, vorticity_map.shape[1])
    Yvec = np.linspace(slatmin, slatmax, vorticity_map.shape[0])
    Xmat, Ymat = np.meshgrid(Xvec, Yvec)
    levels = np.linspace(vmin, vmax, 100)
    contf = axes[kplot].contourf(
        Xmat, Ymat, vorticity_map,
        levels=levels, cmap=cmocean.cm.balance,
        norm=norm, extend='both'
    )
    axes[kplot].set_xlim([slonmin, slonmax])
    axes[kplot].set_ylim([slatmin, slatmax])
    axes[kplot].set_title(f"Cluster {kplot + 1} ({probabilidad:.2f}%)", fontsize=10)

cbar = fig.colorbar(
    contf, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1
)
cbar.set_label(r'$\zeta$ (s$^{-1}$) $\times 10^{-5}$', fontsize=12)
cbar.set_ticks(np.linspace(vmin, vmax, 5))
cbar.ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'{x * 1e5:.1f}')
)
plt.show()

#--------------MAPA DE CALOR---------------------------------------------------------
plt.figure(figsize=(8, 5))
ax = sns.heatmap(
    frecuencia_clusters, cmap='Greys', annot=False,
    cbar_kws={'label': 'Frecuencia'}, linewidths=0.5
)
ax.set_xlabel('Mes')
ax.set_ylabel('Clúster')
ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
ax.set_yticklabels([f'{i + 1}' for i in range(nclust)])
plt.title('Frecuencia de Clúster por Mes', fontsize=16)
plt.tight_layout()
plt.show()
