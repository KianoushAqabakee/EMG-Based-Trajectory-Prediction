close all;
clear all;
clc;
%%

X.train=[];
Y.train=[];
X.test=[];
Y.test=[];

%Speed=[0.3,0.4,0.5,0.6,0.85,0.85,0.6,0.5,0.4,0.3];
index_counter.Train=[];
index_counter.Test=[];
for i = 1:7
    for j = 1:10
        %disp(j)
        %label=['Data',num2str(i),'_speed',num2str(j)];
        label=['Data',num2str(i),'_speed',num2str(j)];%,'_Normalization'
        ss=load([label,'.mat']).([label]);
        [x,y]=dataEx(ss);
        %sp=ones([length(x),1])*Speed(j);
        %x=[x,sp];
        
        x1=x(1:floor(0.8*length(x)),:);
        x2=x(floor(0.8*length(x))+1:end,:);
        y1=y(1:floor(0.8*length(y)),:);
        y2=y(floor(0.8*length(y))+1:end,:);
        
        X.train=[X.train',x1']';
        Y.train=[Y.train',y1']';
        X.test=[X.test',x2']';
        Y.test=[Y.test',y2']';
        
        index_counter.Train=[index_counter.Train,size(x1,1)];
        index_counter.Test=[index_counter.Test,size(x2,1)];
    end
end
%save('X.mat','X')
%save('Y.mat','Y')
%%