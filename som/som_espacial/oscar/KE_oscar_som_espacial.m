clear all 
close all 

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
load C:\LAURA\SOM\urbano_kmeans\urbano_kmeans\urbano_kmeans\wetransfer_kmeans_2022-08-08_1904\map.mat;
load C:\LAURA\SOM\urbano_kmeans\urbano_kmeans\urbano_kmeans\wetransfer_kmeans_2022-08-08_1904\timeglorys.mat

dx = 0.0833;
nx = 33; ny = 53;
[xx, yy] = meshgrid(vlon, vlat);
tt2 = datevec(timeup);

load C:\LAURA\kemans_23anos\OSCAR\K\EKE_oscar.mat
%whos; 

mke = EKE;
clear interm energia

 m = isnan(mke(:,1)); a = find(m==1); b = find(m==0);
 mke(a,:)=[];
 
 M = mke';
 sD = som_data_struct(M); 
 sD = som_normalize(sD,'var');
 rows=3;
 columns=3;

rng(1); % Fijar la semilla aleatoria para reproducibilidad OJOMUY IMPORTANTE
sM = som_randinit(sD,'msize',[columns rows]);
sM=som_impbatch(sM,sD,'msize',[columns rows],'tracking',3,'trainlen',10,'radius',[10 0.1],'lattice','hexa','shape','sheet','neigh','gaussian');
sD = som_denormalize(sD);
sM = som_denormalize(sM);
[bmus,qerrs] = som_bmus(sM,sD,1);
nunits=sM.topol.msize(1,1)*sM.topol.msize(1,2);
histo_ocurrencia=hist(bmus,nunits)
prob_ocurrencia=(histo_ocurrencia/length(bmus))*100

modos = zeros(rows*columns,nx*ny); modos(:,a) = NaN;
modos(:,b) = sM.codebook;

m_proj('mercator','longitudes',[min(double(vlon)) max(double(vlon))],'latitudes',[min(double(vlat)) max(double(vlat))]);

modos_corregido = zeros(size(modos));

for i = 1:size(modos, 1) % Iterar sobre los BMUs
    % Recuperar el mapa en 2D
    modo_actual = reshape(modos(i, :), ny, nx); % Convertir a 2D
    
    % Aplicar las transformaciones necesarias
    modo_actual = rot90(modo_actual, -1); % Rotar 90 grados antihorario
    modo_actual = flip(modo_actual, 2);  % Voltear a lo largo del eje horizontal
    
    % Reaplanar el mapa corregido y almacenar
    modos_corregido(i, :) = modo_actual(:);
end
%-
fontsize = 14;
subplot = @(m,n,p) subtightplot (m, n, p, [0.005 0.005], [0.1 0.01], [0.05 0.05]);
f200 = figure(200); clf(200);
f200.Units = 'normalized';
f200.Position(3) = 0.6881;
f200.Position(4) = 0.8781;
for irow = 1 : rows
    for icol = 1 : columns
        isubplot = columns * (irow - 1) + icol
        subplot(rows, columns, isubplot);
        
        it = isubplot;
        
        m_pcolor(xx,yy,reshape(modos_corregido(it,:),nx,ny));
        %colormap(map)
        colormap(map)
        hold on 
        m_usercoast('claudia','patch',[0.7 0.8 0.9]);
        
        ax = gca;
        clim = round(1000*mean(abs(quantile(modos_corregido(:), [0.05,0.95]))))/1000;
          caxis([0.02, 0.25]);
        if (irow == rows) && (icol == 1)
            m_grid('fontsize', fontsize);
        else
            if irow == rows
                m_grid('fontsize', fontsize, 'yticklabels', {});
            else
                if icol == 1
                    m_grid('fontsize', fontsize, 'xticklabels', {});
                else
                    m_grid('fontsize', fontsize, 'xticklabels', {}, 'yticklabels', {});
                end
            end
        end
        
        frac_x = 0.72;
        pos_x  = xlim; pos_x = pos_x(1) + frac_x * (pos_x(2) - pos_x(1));
        %
        frac_y = 0.2;
        pos_y  = ylim; pos_y = pos_y(1) + frac_y * (pos_y(2) - pos_y(1));
        %
        text(pos_x, pos_y, ['P', num2str(isubplot)], 'FontSize', 20, 'FontWeight', 'bold');
        
        frac_x = 0.61;
        pos_x  = xlim; pos_x = pos_x(1) + frac_x * (pos_x(2) - pos_x(1));
        %
        frac_y = 0.1;
        pos_y  = ylim; pos_y = pos_y(1) + frac_y * (pos_y(2) - pos_y(1));
        %
        text(pos_x, pos_y, ['prob = ', num2str(round(10*prob_ocurrencia(isubplot))/10), ' %'], 'FontSize', fontsize, 'FontWeight', 'bold');
        
        
    end
end
ax = gca;
c = colorbar(ax, 'Location', 'southoutside', 'FontSize', 18);
c.Position = [0.15 0.035 0.7 0.025];
annotation('textbox', [c.Position(1) + c.Position(3) + 0.03, c.Position(2), 0, 0], 'string', 'cm', 'FontSize', 20);
%}

for k = 1:size(M,1);
    h_temporal = som_hits(sM,sD.data(k,:));
    evolution_best_match(k)=find(h_temporal==1);
end
evolution_best_m = evolution_best_match;
% Fecha inicial y final
fecha_inicial = datenum(2000, 1, 1); % 01/01/2000
fecha_final = datenum(2023, 12, 31); % 31/12/2023
nfecha = fecha_inicial:fecha_final; 
vecfecha = datevec(nfecha); 

save('data_eke_somoscar.mat', 'evolution_best_match', 'modos');