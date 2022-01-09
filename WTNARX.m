clc
clear all
close all

load('datahel')
load('datajam')
load('jamtem')
load('heltem')

N=8759;


%
IN=[jamtem(1:N,1)];
OUT=[datajam(1:N,1)+datajam(1:N,2)]/1000;

IN2=[heltem(1:N,1)];
OUT2=[datahel(1:N,1)+datahel(1:N,2)]/1000;

n=1

[C,L] = wavedec(OUT,n,'dmey');
A1 = wrcoef('a',C,L,'dmey',n);

[C,L] = wavedec(OUT2,n,'dmey');
A2 = wrcoef('a',C,L,'dmey',n);


R1=OUT-A1;
R2=OUT2-A2;

S=24;

IN11=IN(1:end-S,:);
IN22=IN(1:end-S,:);
A11=A1(S+1:end,:);
A22=A2(S+1:end,:);
R11=R1(S+1:end,:);
R22=R2(S+1:end,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%APPROXIMATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%APPROXIMATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%APPROXIMATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%APPROXIMATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%APPROXIMATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%APPROXIMATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N=15;

X = tonndata(IN11,false,false);
T = tonndata(A11,false,false);


trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

inputDelays = 1:N;
feedbackDelays = 1:N;
hiddenLayerSize = 10;
net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);


[x,xi,ai,t] = preparets(net,X,{},T);

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

[net,tr] = train(net,x,t,xi,ai);

y = net(x,xi,ai);
e = gsubtract(t,y);
performance = perform(net,t,y);

view(net);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%APPROXIMATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%APPROXIMATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%APPROXIMATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%APPROXIMATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%APPROXIMATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%APPROXIMATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


X = tonndata(IN11,false,false);
T = tonndata(R11,false,false);


trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

inputDelays = 1:N;
feedbackDelays = 1:N;
hiddenLayerSize = 10;
netr = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);


[x,xi,ai,t] = preparets(netr,X,{},T);

netr.divideParam.trainRatio = 70/100;
netr.divideParam.valRatio = 15/100;
netr.divideParam.testRatio = 15/100;

[netr,tr] = train(netr,x,t,xi,ai);

y = netr(x,xi,ai);
e = gsubtract(t,y);
performance = perform(netr,t,y);

view(netr);




% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
X = tonndata(IN22,false,false);
T = tonndata(A22,false,false);
[x1,xi1,ai1,t1] = preparets(net,X,{},T);
y = net(x1,xi1,ai1);

y=cell2mat(y);
y=y';

A22=A22(N+1:end,:);


X = tonndata(IN22,false,false);
T = tonndata(R22,false,false);
[x2,xi2,ai2,t2] = preparets(netr,X,{},T);
yr = netr(x2,xi2,ai2);

yr=cell2mat(yr);
yr=yr';

R22=R22(N+1:end,:);

OUT22new=A22+R22;
model=yr+y;

WTNARXX=[model(300:600,:) OUT22new(300:600,:)];


MSE=mse(OUT22new,model)
R=corr(OUT22new,model)

%% PLOTING
CV=size(model,1);
model=model(1:CV,:);
OUT22new=OUT22new(1:CV,:);





ax1=subplot (1,1,1)
plot(model,'--',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor',[0.5,0.5,0.5])
hold on
plot(OUT22new,'-',...
    'LineWidth',2,...
    'MarkerSize',4,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor',[0.5,0.5,0.5])


hold off
xlim(ax1,[400 800])
% ylim(ax1,[0 3])
% title('WT-NARX prediction','FontSize',20,'FontWeight','bold','Color','k')
ylabel('Heat demand(kW)','FontSize',20,'FontWeight','bold','Color','k')
xlabel('Time (h)','FontSize',20,'FontWeight','bold','Color','k')
set(gca,'FontSize',20,'FontWeight','bold');
legend('WT-NARX model','Ref. tool (IDA-ICE)','Location','northoutside','Orientation','horizontal')
hold on



grid on
grid minor
% ax = gca;
% set(gca,'LineWidth',2)
% ax.GridLineWidth = 2
% ax.GridColor = [0 0 0];
% ax.GridLineStyle = '--';
% ax.MinorGridAlpha = 'bold';
