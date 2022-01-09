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
OUT11=OUT(S+1:end,:);
OUT22=OUT2(S+1:end,:);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%APPROXIMATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%APPROXIMATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%APPROXIMATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%APPROXIMATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%APPROXIMATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%APPROXIMATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N=23;

X = tonndata(IN11,false,false);
T = tonndata(OUT11,false,false);


trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

inputDelays = 1:N;
feedbackDelays = 1:N;
hiddenLayerSize = 15;
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


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
X = tonndata(IN22,false,false);
T = tonndata(OUT22,false,false);
[x1,xi1,ai1,t1] = preparets(net,X,{},T);
y = net(x1,xi1,ai1);

y=cell2mat(y);
y=y';

OUT22=OUT22(N+1:end,:);

for i=1:size(y,1)
    if y(i,1)<0
        y(i,1)=0;
    end
end

        

OUT22new=OUT22;
model=y;

NARXX=y(348:648,:);

%%
ax1=subplot (1,1,1)
plot(model,'--',...
    'LineWidth',2,...
    'MarkerSize',4,...
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
% title('NARX model prediction','FontSize',20,'FontWeight','bold','Color','k')
ylabel('Heat demand(kW)','FontSize',20,'FontWeight','bold','Color','k')
xlabel('Time (h)','FontSize',20,'FontWeight','bold','Color','k')
set(gca,'FontSize',20,'FontWeight','bold');
legend('B-NARX model','Ref. tool (IDA-ICE)','Location','northoutside','Orientation','horizontal')
hold on
grid on
grid minor


figure

subplot(2,1,1)
plot(OUT2(1:70,:))
subplot(2,1,2)
plot(OUT22new(1:50,:))