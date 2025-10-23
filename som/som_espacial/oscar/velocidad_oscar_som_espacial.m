
%%%%%START OF USER-MODIFIABLE SECTION%%%%%%%%%%%% OSCAR SOM

clear all 
close all 
%Define temporal parameters:
seasons='Jan';  %start
midmon ='Jun';  %middle month
seasone='Dec';  %end
yeari=2000; %first year 
yeare=2023; %last year  
%Define spatial parameters:
%sdomain (for clusters)
slonmin=-84;
slonmax=-70;
slatmin=7;
slatmax=16;
%bdomain (for plotting)
blonmin=-84;
blonmax=-70;
blatmin=7;
blatmax=16;

% Generar timeup (fechas en formato MATLAB)
start_date = datetime(2000, 1, 1);
end_date = datetime(2023, 12, 31);
fechas = start_date:end_date; % Crear rango de fechas
timeup = datenum(fechas); % Convertir fechas a formato numérico de MATLAB

addpath ('C:\LAURA\SOM\urbano_kmeans\urbano_kmeans\urbano_kmeans\wetransfer_kmeans_2022-08-08_1904\somtoolbox');
addpath ('C:\LAURA\PROGRAMAS\m_map');
addpath('C:\LAURA\SOM\urbano_kmeans\urbano_kmeans\urbano_kmeans\wetransfer_kmeans_2022-08-08_1904');
data = load('C:/LAURA/oscar_combined.mat');
vlat = data.lat;
vlon = data.lon;
 
dx = 0.0833;
nx = 33; ny = 53;
[xx, yy] = meshgrid(vlon, vlat);
tt2 = datevec(timeup);

% Cargar datos de las componentes u y v
u = load('C:\LAURA\OSCAR1cuarto\U_oscar2final.mat');
v = load('C:\LAURA\OSCAR1cuarto\V_oscar2final.mat');
up = u.u; % Componente u
vp = v.v; % Componente v
%whos; 
up2=up;vp2=vp;

% Validar dimensiones de u y v
assert(isequal(size(up), size(vp)), 'Las dimensiones de u y v no coinciden.');
up1=up;vp1=vp;
disp('Circulation variable (currents) has been read and stored.');

% Combinar las componentes u y v en una matriz de características
uv = [up; vp]; % Transponer y combinar (2 características por fila)
mke = uv;
clear interm energia

 m = isnan(mke(:,1)); a = find(m==1); b = find(m==0);
 mke(a,:)=[];
%%%%%END OF USER-MODIFIABLE SECTION%%%%%%%%%%%%

%% Aplicar SOM
%addpath('C:\Users\lortegac\LAURA\SOMToolbox'); % Ruta al toolbox de SOM

% Crear estructura de datos para SOM
 M = mke'; % Filas = muestras, columnas = características (u y v)
sD = som_data_struct(M); % Crear estructura de datos SOM
sD = som_normalize(sD, 'var'); % Normalizar las variables
% Justo después de crear y normalizar la estructura sD:
sD.data = double(sD.data);  % Convertir a double

% Configurar el SOM
rows = 3; % Número de filas en la malla SOM
columns = 3; % Número de columnas en la malla SOM
sM = som_randinit(sD, 'msize', [columns rows]); % Inicializar SOM
sM=som_impbatch(sM,sD,'msize',[columns rows],'tracking',3,'trainlen',10,'radius',[10 0.1],'lattice','hexa','shape','sheet','neigh','gaussian');
sD = som_denormalize(sD);
sM = som_denormalize(sM);
% Obtener las unidades ganadoras (BMUs)
[bmus, qerrs] = som_bmus(sM, sD, 1);
nunits=sM.topol.msize(1,1)*sM.topol.msize(1,2);
histo_ocurrencia=hist(bmus,nunits)
prob_ocurrencia=(histo_ocurrencia/length(bmus))*100;


% Extrae dimensiones del SOM
rows = sM.topol.msize(2);
columns = sM.topol.msize(1);
nunits = rows * columns; 

modos = zeros(rows*columns,nx*ny); modos(:,a) = NaN;
modos(:,b) = sM.codebook;

uqcompa = modos(:, 1:1749); % Tomamos la primera mitad
vqcompa = modos(:, 1749+1:end); % Tomamos la segunda mitad

uqcompa2=permute(reshape(uqcompa,nunits,53,33),[1,2,3]);
vqcompa2=permute(reshape(vqcompa,nunits,53,33),[1,2,3]);
%
labs = {'(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)'};

%Con PCOLOR para uqcompa2 gráficas con líneas de corrientes (Streamlice)
Xmat=yy;
Ymat=xx;

Xmat=Xmat';Ymat=Ymat';
%Se crea una lista donde se almacenarán los mapas de magnitud de velocidad para cada cluster
vemap_list = cell(1, nunits);
for kplot = 1:nunits
    umap = squeeze(uqcompa2(kplot, :, :));  %extraigo los componentes medios de velocidades en diracciones u
    vmap = squeeze(vqcompa2(kplot, :, :));  %extraigo los componentes medios de velocidades en diracciones v
    vemap_list{kplot} = sqrt(umap.^2 + vmap.^2);    %Cálculo de la magnitud de la velocidad raiz(X^2+Y^2)
end

valor_maximo_global=-inf;
figure;
for kplot = 1:nunits
    % Extraer vemap precalculado
    vemap = vemap_list{kplot};
    %Calcular valor maaximo
    valor_maximo =max(vemap(:));
    fprintf('Valor maximo en el mapa %d: %f\n', kplot, valor_maximo');
    if valor_maximo > valor_maximo_global
        valor_maximo_global=valor_maximo;
    end
    
    subplot(rows, columns, kplot);

    pcolor(Xmat, Ymat, vemap');
    shading interp;
    hold on;
    
    hs = streamslice(Xmat, Ymat, squeeze(uqcompa2(kplot, :, :))', squeeze(vqcompa2(kplot, :, :))', 2);
    set(hs, 'Color', [0 0 0], 'LineWidth', 1);
    
    % Dibuja las líneas de costa
    %plot(line2000_shore(:, 1), line2000_shore(:, 2), 'k');
    
    xlim([slonmin slonmax]);
    ylim([slatmin slatmax]);
    caxis([0 0.87]);
    box on;
  
    title([labs{kplot} ' Visuallización de datos ' num2str(kplot)]);
    
    colorbar;
    ylabel(colorbar, 'Velocidad [m/s]'); 
end



for k = 1:size(M,1);
    h_temporal = som_hits(sM,sD.data(k,:));
    evolution_best_match(k)=find(h_temporal==1);
end
evolution_best_m = evolution_best_match;

save('data_corrientes_somoscar.mat', 'evolution_best_match', 'uqcompa', 'vqcompa');
