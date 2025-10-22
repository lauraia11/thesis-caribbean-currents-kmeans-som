%%%%% START OF USER-MODIFIABLE SECTION %%%%%

clear all
close all

% Parámetros temporales
seasons = 'Jan';
midmon  = 'Jun';
seasone = 'Dec';
yeari   = 2000;
yeare   = 2023;

% Parámetros espaciales
slonmin = -84; slonmax = -70;
slatmin = 7;   slatmax = 16;
blonmin = -84; blonmax = -70;
blatmin = 7;   blatmax = 16;

addpath('C:\LAURA\SOM\urbano_kmeans\urbano_kmeans\urbano_kmeans\wetransfer_kmeans_2022-08-08_1904\somtoolbox');
addpath('C:\LAURA\PROGRAMAS\m_map');
addpath('C:\LAURA\SOM\urbano_kmeans\urbano_kmeans\urbano_kmeans\wetransfer_kmeans_2022-08-08_1904');

load C:\LAURA\SOM\urbano_kmeans\urbano_kmeans\urbano_kmeans\wetransfer_kmeans_2022-08-08_1904\map.mat
load C:\LAURA\SOM\urbano_kmeans\urbano_kmeans\urbano_kmeans\wetransfer_kmeans_2022-08-08_1904\lonur.mat
load C:\LAURA\SOM\urbano_kmeans\urbano_kmeans\urbano_kmeans\wetransfer_kmeans_2022-08-08_1904\latur.mat
load C:\LAURA\SOM\urbano_kmeans\urbano_kmeans\urbano_kmeans\wetransfer_kmeans_2022-08-08_1904\timeglorys.mat

dx = 0.0833;
nx = 169; ny = 109;
xx = reshape(vlon, nx, ny);
yy = reshape(vlat, nx, ny);

lonu = load('C:\LAURA\Patronesclimaticos\lonur.mat');
latu = load('C:\LAURA\Patronesclimaticos\latur.mat');
lonu = lonu.vlon;
latu = latu.vlat;

u = load('C:\LAURA\NUEVOS_DATASE_2018-2023\filtrado_Caribe\u_caribe_actual.mat');
v = load('C:\LAURA\NUEVOS_DATASE_2018-2023\filtrado_Caribe\v_caribe_actual.mat');
up = u.u;
vp = v.v;

start_date_dataset = datetime(1993,1,1);
start_date = datetime(2000,1,1);
end_date   = datetime(2023,12,31);

days_from_start_to_2000 = daysact(start_date_dataset, start_date);
days_from_start_to_2023 = daysact(start_date_dataset, end_date);
mask = days_from_start_to_2000:days_from_start_to_2023;
up = up(:, mask);
vp = vp(:, mask);

up2 = up; vp2 = vp;
speed = sqrt(up2.^2 + vp2.^2);

assert(isequal(size(up), size(vp)), 'Las dimensiones de u y v no coinciden.');
up1 = up; vp1 = vp;
disp('Circulation variable (currents) has been read and stored.');

uv = [up; vp];
mke = uv;
m = isnan(mke(:,1)); a = find(m==1); b = find(m==0);
mke(a,:) = [];

%%%%% END OF USER-MODIFIABLE SECTION %%%%%

%% Aplicar SOM
M = mke';
sD = som_data_struct(M);
sD = som_normalize(sD,'var');
sD.data = double(sD.data);

rows = 3; columns = 3;
sM = som_randinit(sD,'msize',[columns rows]);
sM = som_impbatch(sM, sD, 'msize',[columns rows], 'tracking',3, ...
    'trainlen',10,'radius',[10 0.1],'lattice','hexa','shape','sheet','neigh','gaussian');
sD = som_denormalize(sD);
sM = som_denormalize(sM);

[bmus, qerrs] = som_bmus(sM, sD, 1);
nunits = sM.topol.msize(1,1) * sM.topol.msize(1,2);
histo_ocurrencia = hist(bmus, nunits);
prob_ocurrencia = (histo_ocurrencia / length(bmus)) * 100;

rows = sM.topol.msize(2);
columns = sM.topol.msize(1);
nunits = rows * columns;

modos = zeros(rows * columns, nx * ny);
modos(:, a) = NaN;
modos(:, b) = sM.codebook;

uqcompa = modos(:, 1:18421);
vqcompa = modos(:, 18421+1:end);

uqcompa2 = permute(reshape(uqcompa, nunits, 169, 109), [1,2,3]);
vqcompa2 = permute(reshape(vqcompa, nunits, 169, 109), [1,2,3]);

labs = {'(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)'};

Xmat = reshape(lonu,169,109)';
Ymat = reshape(latu,169,109)';

vemap_list = cell(1, nunits);
for kplot = 1:nunits
    umap = squeeze(uqcompa2(kplot,:,:));
    vmap = squeeze(vqcompa2(kplot,:,:));
    vemap_list{kplot} = sqrt(umap.^2 + vmap.^2);
end

valor_maximo_global = -inf;
figure;
for kplot = 1:nunits
    vemap = vemap_list{kplot};
    valor_maximo = max(vemap(:));
    fprintf('Valor máximo en el mapa %d: %f\n', kplot, valor_maximo);
    if valor_maximo > valor_maximo_global
        valor_maximo_global = valor_maximo;
    end

    subplot(rows, columns, kplot);
    pcolor(Xmat, Ymat, vemap'); shading interp; hold on;
    hs = streamslice(Xmat, Ymat, squeeze(uqcompa2(kplot,:,:))', squeeze(vqcompa2(kplot,:,:))', 2);
    set(hs, 'Color',[0 0 0],'LineWidth',1);

    xlim([slonmin slonmax]);
    ylim([slatmin slatmax]);
    caxis([0 0.87]);
    box on;
    title([labs{kplot} ' Visualización de datos ' num2str(kplot)]);
    colorbar; ylabel(colorbar, 'Velocidad [m/s]');
end

for k = 1:size(M,1)
    h_temporal = som_hits(sM, sD.data(k,:));
    evolution_best_match(k) = find(h_temporal == 1);
end

fecha_inicial = datenum(2000,1,1);
fecha_final = datenum(2023,12,31);
nfecha = fecha_inicial:fecha_final;
vecfecha = datevec(nfecha);

ocurr = zeros(rows * columns, 12);
for tt = 1:12
    ind = find(vecfecha(:,2) == tt);
    for pp = 1:(rows * columns)
        ocurr(pp,tt) = length(find(evolution_best_match(ind) == pp));
    end
end

ocurr2 = 100 * ocurr ./ length(nfecha);
ocurr2(:,13) = sum(ocurr2,2);
ocurr2(rows * columns + 1, :) = sum(ocurr2,1);

figure;
total_mods = rows * columns;

for i = 1:(total_mods - 2)
    plot(1:12, ocurr2(i,1:12), 'LineWidth', 2); hold on;
end
for i = (total_mods - 1):total_mods
    plot(1:12, ocurr2(i,1:12), 'LineWidth', 2, 'LineStyle', ':'); hold on;
end

legend(arrayfun(@(x) ['Neu', num2str(x)], 1:total_mods, 'UniformOutput', false));
ylabel('% Frecuencia');
xlabel('Meses');
axis tight; grid on;
