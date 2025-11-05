from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import cv2

def stand(y, opt):
    my = np.nanmean(y, axis=0)
    
    sty = np.nanstd(y, axis=0,ddof=1)
   # Evitar divisiones por desviaciones estándar muy pequeñas
    sty[sty < 1e-8] = 1.0  # Umbral mínimo para la desviación estándar
    my = np.tile(my,  (y.shape[0], 1))
    sty = np.tile(sty, (y.shape[0], 1))
    
    # opt='m'
    if opt == 'm':
            x = y - my
    elif opt =='s':
        x = (y - my) / sty
    return x

#vERIFIACAR FUNCION STAND

def kmeans_ci_py (X, stan, prop, nclus, nsim):
    if stan is not None:
        X = stand(X, stan)
    
    
    U, S, Vt = np.linalg.svd(X,full_matrices=False)
    s = (S)**2
    sc = s/np.sum(s)
    a = np.where(np.cumsum(sc) > prop)
    b = a[0][0]
    X = U[:, :b+1] @np.diag(S[:b+1])
        
    r, c = X.shape
    # print('Time steps and EOF ccouting for the varfrac: ')
    # print(X.shape)
    
    k = np.zeros((r,nsim), dtype=int)
    MC = np.zeros((nclus * c, nsim))
    # nsim=10
    for i in range(nsim):
        # print(i)
        kmeans = KMeans(n_clusters=nclus,init='k-means++', max_iter=10, n_init=10)
        # algoritmo = KMeans(n_clusters = a,  init = 'k-means++',max_iter = 300, n_init = 100, random_state=0)
        kmeans.fit(X)

        k[:,i] = kmeans.predict(X)
        mean_cluster = kmeans.cluster_centers_
        # print(k[:,i])
        # mean_cluster = np.zeros((nclus,c))
        
        # for j in range(nclus):
        #     mean_cluster[j,:] = np.mean(X[np.flatnonzero(k[:,i] == j), :], axis=0)
        #     # ass=X[np.flatnonzero(k[:,i] == j), :]
           
        mean_cluster2 = stand(mean_cluster.T, 's') 
        
        aux=cv2.flip(mean_cluster2.T,0)
        MC[:, i] = aux.flatten()
            
    ACCmax = np.zeros((nclus, nsim))
            
    for i in range(nclus):
        for j in range(nsim):
            sample1 = MC[(i * c):(i + 1)* c, j]
            a=np.where(np.arange(nsim) !=j)[0]
            sample2 = MC[:, a].reshape(c, (nsim - 1) * nclus,order='F')
            ACC = (1 / (c - 1)) * np.dot(sample1.T, sample2)   
            ACC = ACC.reshape(nclus, nsim - 1,order='F')
            ACCmax[i, j] = np.min(np.max(ACC, axis=0))
    
    mean_ACCmax = np.mean(ACCmax, axis=0)
    # Selección de la mejor partición
    part = np.where(mean_ACCmax == np.max(mean_ACCmax))[0]
    CI = np.mean(ACCmax)
    if len(part) >= 1:
            part = part[0]
    return CI, k[:, part]