close all;
clear all;
clc;
%%

X.train=[];
Y.train=[];
X.test=[];
Y.test=[];

%Speed=[0.3,0.4,0.5,0.6,0.85,0.85,0.6,0.5,0.4,0.3];
%Data7_speed1_Normalization
for i = 1:7
    for j = 1:10
        %disp(j)
        label=['Data',num2str(i),'_speed',num2str(j)];
        label2=[label,'_Normalization'];
        ss=load([label,'.mat']).([label]);
        x=dataEx2(ss);
        assignin('base',label2,x)
        save([label2,'.mat'],label2)
    end
end
%save('X_s1_l.mat','X')
%save('Y_s1_l.mat','Y')
%%