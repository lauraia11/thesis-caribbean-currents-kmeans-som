function [x]=stan(y,opt,missing);

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


