function data=normalizeData(data)
[N,D]=size(data);
maxD=max(data,[],1);
minD=min(data,[],1);
maxMinD=maxD-minD;
maxMinD=repmat(maxMinD,N,1);
data=data-repmat(minD,N,1);
data=data./maxMinD;
end