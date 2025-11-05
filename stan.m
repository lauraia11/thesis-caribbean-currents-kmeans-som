function [x]=stan(y,opt,missing);

% [x]=stan(y,opt,missing);
%
% This function standardizes the columns of an input matrix to zero mean
% and eventually unit variance. The missing values are not filled. 
% Use nanstan.m if you want to replace missing values by the long-term mean.
%
% Input
% 'y' : matrix of real number to be standardized
% 'opt' : character string ='m' (output data are normalized to zero mean)
% and ='s' (output data are standardized to zero mean and unit variance). 
% 'missing' : scalar defining the missing value (if missing = NaN, it is
% not necessary to define missing).
% 
% Output
% 'x' : matrix of standardized data. The missing data are coded into NaN.
%
% Vincent Moron
% March 1996

if nargin==2;
    missing=[NaN];
else
    y(find(y==missing))=NaN*ones(size(find(y==missing)));
end

[n,c]=size(y);
my=nan_mean(y);
sty=nan_std(y);
my=ones(n,1)*my;
sty=ones(n,1)*sty;
if strcmp(opt,'m')
   x=(y-my);
else
	x=(y-my)./sty;
end;
