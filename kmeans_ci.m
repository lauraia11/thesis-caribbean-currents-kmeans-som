

function [CI,K, centroids]=kmeans_ci(X,stand,prop,nclus,nsim);

% [CI,K]=kmeans_ci(X,stand,prop,nclus,nsim);
%
%  
%
% Input
% 'X': input matrix to be classified (rows are classified)
% 'stand': option for standardising the matrix (by columns) = 'm' for just
% removing the long-term mean and ='s' for standardising to zero mean and
% unit variance. if stand is empty, the matrix is not standardized.
% 'prop': scalar giving the proportion of variance to be retained in the
% EOF pre-filtering step. if prop is empty, no EOF-prefiltering is done
% 'nclus': number of cluster
% 'nsim' : number of different partition simulated (typically nsim= 50-100)
%
% Outputs
% 'CI': scalar giving the classifiability index
% 'K': row vector giving the cluster for each row of input matrix 'X'.
%
% ref. Michelangeli et al., JAS, 1995 (1230-1246)
%
 
if ~isempty(stand);
    X=stan(X,stand);
end

if ~isempty(prop);
    [U,S,V]=svd(X,0);
    s=diag(S).^2;
    sc=s/sum(s);
    a=find(cumsum(sc)> prop);
    a=a(1);
    X=U(:,1:a)*S(1:a,1:a);
end

[r,c]=size(X);
disp('Time steps and EOF accounting for the varfrac:')
size(X)
centroids_all = cell(nsim, 1);

for i=1:nsim;
    k(:,i)=kmeans(X,nclus,'Maxiter',1000,'EmptyAction','drop');
    for j=1:nclus;
       mean_cluster(j,:)=mean(X(find(k(:,i)==j),:));
    end
    %Guardar los centroides en la matriz 3D
    centroids_all{i} = mean_cluster;
    
    mean_cluster2=stan(mean_cluster','s'); clear mean_cluster
    MC(:,i)=mean_cluster2(:); % centroids stored in MC matrix
end
centroids = centroids_all{end};
% Anomaly correlation between partition P and Q
for i=1:nclus;
    for j=1:nsim;
        sample1=MC(((i-1)*c)+1:i*c,j);
        a=find(j~=[1:nsim]);
        sample2=reshape(MC(:,a),c,(nsim-1)*nclus); clear ACC
        ACC=(1/(c-1))*sample1'*sample2;
        ACC=reshape(ACC,nclus,nsim-1);
        ACCmax(i,j)=min(max(ACC)); % mean similarity for cluster i of the simulation j
    end
end

part=find(mean(ACCmax)==max(mean(ACCmax)));
CI=mean(mean(ACCmax)); % Classification index for the partition on 'nclus' clusters
if length(part) > 1; part=part(1); end
K=k(:,part); % best partition that maximize the ACC with the nsim-1 other partitions

