%SOM TEMPORAL APLICADO AL CALULO DE VORTICIDAD (CROCO)
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
data = load('C:\LAURA\SOM\SOM_CROCO\resultados_semanalesCROCOtrue_simetricos.mat');
up = data.U_weekly_avg;
vp = data.V_weekly_avg;
%%
%probar
% Obtener dimensiones de las matrices
[nt, nlat, nlon] = size(up);

% Inicializar las matrices para guardar los resultados
u_flattened = zeros(nt, nlat * nlon); 
v_flattened = zeros(nt, nlat * nlon); 

for t = 1:nt
    % Aplanar el espacio lat x lon en un vector fila
    u_flattened(t, :) = reshape(up(t, :, :), 1, []);
    v_flattened(t, :) = reshape(vp(t, :, :), 1, []);
end
%------------
% Transponer para organizar las matrices (espacial x tiempo)
up = u_flattened';
vp = v_flattened';

% Guardar copias originales (por si las necesitamos después)
up2 = up; 
vp2 = vp;
%cargar datos de las componentes pm y pn------------(para tener en cuenta
%la curvatura terrestre)
%Tengo que volver a generar pm y pn ya que estoy manejando promedios
%semanales 
pn = h5read('C:/LAURA/SOM/SOM_CROCO/prom_semanal/pn_croco_2d.mat', '/pn');
pm = h5read('C:/LAURA/SOM/SOM_CROCO/prom_semanal/pm_croco_2d.mat', '/pm');

pm = pm';
pn = pn';

nspatial = size(up, 1); 
ntime = size(up, 2);   

% Inicializar la matriz de vorticidad
vorticity = NaN(nspatial, ntime);

for t = 1:ntime
    try
        % Extraer u, v, pm, pn para el instante t
        u = up(:, t);    
        v = vp(:, t);    
        pm_t = pm(:, t); % (83841, 1)
        pn_t = pn(:, t); % (83841, 1)
        
        % Crear una máscara lógica para ignorar NaNs
        valid_mask = ~isnan(u) & ~isnan(v) & ~isnan(pm_t) & ~isnan(pn_t);
        
        % Crear vectores válidos
        u_valid = u(valid_mask);
        v_valid = v(valid_mask);
        pm_valid = pm_t(valid_mask);
        pn_valid = pn_t(valid_mask);

        % Aplicar circshift solo a los datos válidos
        u_shifted = circshift(u_valid, -1);
        v_shifted = circshift(v_valid, -1);

        % Evitar errores en los bordes (circshift introduce valores inválidos en los extremos)
        u_shifted(end) = NaN;
        v_shifted(end) = NaN;
        % Calcular derivadas espaciales
    	dV_dx = pm_valid .* (v_shifted - v_valid);
        dU_dy = pn_valid .* (u_shifted - u_valid);

        % Calcular vorticidad
        vorticity_t = NaN(size(valid_mask));
        vorticity_t(valid_mask) = dV_dx - dU_dy;

        % Guardar el resultado en la matriz de vorticidad
        vorticity(:, t) = vorticity_t;
    catch ME
        % Capturar y mostrar el error
        fprintf('Error en el cálculo de vorticidad para el día %d: %s\n', t, ME.message);
    end
end
a_vort = find(any(isnan(vorticity), 2)); % Índices con NaNs en u
b_vort = find(~any(isnan(vorticity), 2)); % Índices válidos en u
% Remover filas con NaNs por separado
vorticity(a_vort, :) = [];

tiene_nan = any(isnan(vorticity(:)));
disp(['Vorticidad contiene NaNs: ', num2str(tiene_nan)]);

mke = vorticity; % Transponer nuevamente para formato SOM (características x muestras)

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

rng(1); % Fijar la semilla aleatoria para reproducibilidad OJOMUY IMPORTANTE
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
mapa(a_vort) = NaN;
mapa(b_vort) = evolution_best_match;
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
% CREACION VARIABLE TIEMPO SEMANAL
startDate = datetime(2000, 1, 1);
endDate = datetime(2023, 12, 31);

% Crear un vector de fechas con frecuencia semanal
dates = startDate:calweeks(1):endDate;

% Extraer el mes y el año de cada fecha
months = month(dates);
years = year(dates);

% Calcular el número de semana dentro de cada mes
weeks = zeros(size(dates));
for i = 1:length(dates)
    % Contar semanas dentro del mismo mes y año
    weeks(i) = sum((years == years(i)) & (months == months(i)) & (dates <= dates(i)));
end

% Crear una tabla para mostrar las etiquetas
labelsTable = table(weeks', months', years', 'VariableNames', {'Semana', 'Mes', 'Año'});
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
%%

% Número de clases SOM
n_clases = max(bmus);

% Expandir weekly_mapping para abarcar todos los puntos espaciales
n_puntos = nx * ny; % Total de puntos espaciales en el mapa
weekly_mapping_expanded = repelem(monthly_mapping, n_puntos);

% Inicializar matriz para frecuencias
frecuencia_semanal = zeros(n_clases, 53); % Clases x Semanas (máximo 53 semanas en un año)

% Calcular frecuencias por clase y semana
for clase = 1:n_clases
    for semana = 1:53
        % Contar ocurrencias de la clase SOM en la semana actual
        frecuencia_semanal(clase, semana) = sum(bmus == clase & weekly_mapping_expanded == semana);
    end
end
% Crear el gráfico
figure;
hold on;
c1 = lines(n_clases); % Generar una paleta de colores para cada clase
for clase = 1:n_clases
    plot(1:53, frecuencia_semanal_normalizada(clase, :), '-o', 'LineWidth', 1.5, ...
        'Color', c1(clase, :), 'DisplayName', ['Clase ', num2str(clase)]);
end

% Personalizar el gráfico
set(gca, 'XTick', 1:4:53, 'FontSize', 12, 'FontWeight', 'bold'); % Opcional: Espaciar cada 4 semanas
xlabel('Semanas', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Frecuencia (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('Frecuencia Semanal de Clases SOM', 'FontSize', 14, 'FontWeight', 'bold');
legend('show', 'Location', 'northoutside', 'Orientation', 'horizontal');
grid on;
hold off;
