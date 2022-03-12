
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%loading data%%%%%%%%%%%%%%%%%%%%%
inputData=load('D:\ye\Heterogeneous clustering\coil20_64x64.mat');
x=inputData.fea;
s=inputData.gnd;
data=x;
index=[]
CK=20;
tic;
for i =1:CK
    indexF=find(s==i);
    index=[index;indexF];
end;
xPcaD1=x(index,:);
D1_labels=s(index,:)';

index1=[1:2:72*CK];
index2=[2:2:72*CK];


[N,D]=size(xPcaD1);

plusTemp=ones(1,D)*(-1);
temp=repmat(plusTemp,N,1);
xPcaD2=temp.*xPcaD1;

D2_labels=D1_labels;
xPcaD1=normalizeData(xPcaD1);
xPcaD2=normalizeData(xPcaD2);
% xPcaD2=fliplr(xPcaD2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xPcaD1=xPcaD1(index1,:);
D1_labels=D1_labels(index1);
xPcaD2=xPcaD2(index2,:);
D2_labels=D2_labels(index2);


s=D1_labels+2;
c=s*10;

xPca=xPcaD1;
hold on;
s=D2_labels+2;
c=s*10;

xPca=xPcaD2;

temp=xPcaD1;
xPcaD1=xPcaD2;
xPcaD2=temp;

temp=D1_labels;
D1_labels=D2_labels;
D2_labels=temp;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%compute the similarity%%%%%%%%%%%%%%%%%%%%%

kt=150;
e=10;
options.show = 0;
options.lambda = 0;
options.k = 15;
W = constructW(xPcaD1, options);
W1=full(W);


W = constructW(xPcaD2, options);
W2=full(W);




C3=computeTheInnerSimeShortestPath(xPcaD1,xPcaD2);
C3=full(C3);
WC=[W1,C3];
WC2=[C3',W2];
W=[WC;WC2];
[U,D]=eig(W);

% [engVector engValue]=pca(U);
% xPca=U*engVector(:,1:3);
xPca=U;

label=[D1_labels,D2_labels];
s=label+2;
c=s*10;

data=[xPcaD1;xPcaD2];
save('Wcoil20.mat', 'W', 'label','data');
toc;



