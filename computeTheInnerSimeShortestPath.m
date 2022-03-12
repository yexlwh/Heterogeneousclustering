function [ C2 ] = computeTheInnerSimeShortestPath( xPcaD1,xPcaD2 )
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明

options.show = 0;
options.lambda = 0;
options.k = 8;
W = constructW(xPcaD1, options);
W=full(W);
W1=(1-(W>0))*10;
W=W+W1;


[C1] = FloydWarshall(W);


options.show = 0;
options.lambda = 0;
options.k = 8;
W = constructW(xPcaD2, options);
W=full(W);
W1=(1-(W>0))*10;
W=W+W1;


[C2] = FloydWarshall(W);

% YDst=xPcaD1;
% Ysrc=xPcaD1;
% [NSrc DSrc]=size(Ysrc);
% [N D]=size(YDst);
% C1 = repmat( sum(YDst'.^2)', [1 NSrc] ) + ...
%     repmat( sum(Ysrc'.^2), [N 1] ) - 2*YDst*Ysrc';

% YDst=xPcaD2;
% Ysrc=xPcaD2;
% [NSrc DSrc]=size(Ysrc);
% [N D]=size(YDst);
% C2 = repmat( sum(YDst'.^2)', [1 NSrc] ) + ...
%     repmat( sum(Ysrc'.^2), [N 1] ) - 2*YDst*Ysrc';

[Sc1,index]=sort(C1,2);

[Sc2,index]=sort(C2,2);

[NSc1,DSc1]=size(Sc1);
[NSc2,DSc2]=size(Sc2);
if DSc1<DSc2
    Dsc=DSc1;
else
    Dsc=DSc2;
end;

NSc1=find(Sc1==10);


if isempty(NSc1)
    NSc1=10;
else
    NSc1=int32(NSc1/DSc1);
    NSc1=NSc1(1);
end;

NSc2=find(Sc2==10);

if isempty(NSc2)
    NSc2=10;
else
    NSc2=int32(NSc2/DSc2);
    NSc2=NSc2(1);
end;

if NSc1<NSc2
    Dsc3=NSc1;
else
    Dsc3=NSc2;
end;

if Dsc>Dsc3
    Dsc=Dsc3;
else
    Dsc=Dsc;
end;

Sc1=Sc1(:,1:Dsc);
Sc2=Sc2(:,1:Dsc);
% max1=(max(Sc1));
% max2=(max(Sc2));
% Sc1=Sc1./max1;
% Sc2=Sc2./max2;
YDst=Sc1./repmat(Sc1(:,end),1,double(Dsc));
Ysrc=Sc2./repmat(Sc2(:,end),1,double(Dsc));
[NSrc DSrc]=size(Ysrc);
[N D]=size(YDst);
% C2 = repmat( sum(YDst'.^2)', [1 NSrc] ) + ...
%     repmat( sum(Ysrc'.^2), [N 1] ) - 2*YDst*Ysrc';
C2=pdist2(YDst,Ysrc);

[NSc1,index]=sort(C2,2);

[N,D]=size(C2);
C2=zeros(N,N);
for i=1:N
    C2(i,index(i,1:10))=0.9;
end;
% C2(:,index(:,10:end))=0;


end

