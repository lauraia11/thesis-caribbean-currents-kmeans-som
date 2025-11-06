clear all 
close all 
dx = 0.0833;

ensayo = load('C:\LAURA\CROCO\lon_lat.mat');
% Aplanar las matrices en vectores columna
lat = ensayo.lat;
lon = ensayo.lon;
vlat = reshape(lat', [], 1); % Transponer y luego convertir en columna
vlon = reshape(lon', [], 1); % Transponer y luego convertir en columna

nx = 469; ny = 295;
xx = reshape(vlon,nx,ny); yy = reshape(vlat,nx,ny);


%cargar datos de las componentes u y v--------------------------
data = load('C:\LAURA\CROCO\resultados_semanalesCROCOtrue.mat');
up = data.U_weekly_avg;
vp = data.V_weekly_avg;
%%
%probar
% Obtener dimensiones de las matrices
[nt, nlat, nlon] = size(up);

% Inicializar las matrices para guardar los resultados
u_flattened = zeros(nt, nlat * nlon); % Matriz 2D para U
v_flattened = zeros(nt, nlat * nlon); % Matriz 2D para V

% Aplanar cada paso temporal
for t = 1:nt
    % Aplanar el espacio lat x lon en un vector fila
    u_flattened(t, :) = reshape(up(t, :, :), 1, []); % Convierte a vector fila
    v_flattened(t, :) = reshape(vp(t, :, :), 1, []); % Convierte a vector fila
end
%------------
up = u_flattened;
vp = v_flattened;

% Calcular la magnitud de la velocidad
velocidad = sqrt(up.^2 + vp.^2); % up y vp son (nlat*nlon x nt)
velocidad = velocidad';

a_velocidad = find(any(isnan(velocidad), 2)); % Índices con NaNs en u
b_velocidad = find(~any(isnan(velocidad), 2)); % Índices válidos en u
% Remover filas con NaNs por separado
velocidad(a_velocidad, :) = [];

tiene_nan = any(isnan(velocidad(:)));
disp(['velocidad contiene NaNs: ', num2str(tiene_nan)]);

mke = velocidad; % Transponer nuevamente para formato SOM (características x muestras)

%----------------------------------------------------------------
addpath('C:\LAURA\SOM\urbano_kmeans\urbano_kmeans\urbano_kmeans\wetransfer_kmeans_2022-08-08_1904\somtoolbox'); % Ruta al toolbox de SOM

%mke(a_u,:)=[];
 M = mke;
 % Verificar si M contiene algún NaN
tiene_nan = any(isnan(M(:)));
disp(['M contiene NaNs: ', num2str(tiene_nan)]);
 sD = som_data_struct(M); 
 sD = som_normalize(sD,'var');
rows=3;
columns=1;
 
sM = som_randinit(sD,'msize',[columns rows]);
sM=som_impbatch(sM,sD,'msize',[columns rows],'tracking',3,'trainlen',10,'radius',[10 0.1],'lattice','hexa','shape','sheet','neigh','gaussian');
sD = som_denormalize(sD);
sM = som_denormalize(sM);
[bmus,qerrs] = som_bmus(sM,sD,1);
nunits=sM.topol.msize(1,1)*sM.topol.msize(1,2);
histo_ocurrencia=hist(bmus,nunits)
prob_ocurrencia=(histo_ocurrencia/length(bmus))*100

modos =  sM.codebook;
for k = 1:size(M,1);
    h_temporal = som_hits(sM,sD.data(k,:));
    evolution_best_match(k)=find(h_temporal==1);
end

mapa = zeros(nx*ny,1);
mapa(a_velocidad) = NaN;
mapa(b_velocidad) = evolution_best_match;
c1 = distinguishable_colors(rows*columns);
figure(300)
m_proj('mercator','longitudes',[min(double(vlon)) max(double(vlon))],'latitudes',[min(double(vlat)) max(double(vlat))]);
m_pcolor(xx,yy,reshape(mapa,nx,ny))
    m_usercoast('claudia','patch',[0.7 0.8 0.9]);
    m_grid('fontsize', 20);
 colormap(c1)
caxis([0.5 rows*columns+0.5])
ct=colorbar;
ct.Ticks = [1:rows*columns];
%%
%CONVERSIÓN A GEOTIFF
nx = 469; ny = 295;
mapa_2d = reshape(mapa, nx, ny); % Convierte el mapa lineal a una matriz 2D
mapa_2d = rot90(mapa_2d); % Rotar 90° a la izquierda
mapa_2d = flipud(mapa_2d);

R = georasterref('RasterSize', size(mapa_2d), ...
                 'LatitudeLimits', [min(double(vlat)), max(double(vlat))], ...
                 'LongitudeLimits', [min(double(vlon)), max(double(vlon))]);


geotiffwrite('mapa_resultado_tiff.tif', mapa_2d, R);
