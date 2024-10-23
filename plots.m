%% stampe dei grafici

function plotModel(t_true,yTrain)
    tiledlayout(1,2);
    nexttile
    plot(t_true,yTrain(1,:),'k-')
    hold on
    plot(t_true,yTrain(2,:),'r-')
    hold off
    xlabel('$t$','Interpreter','latex')
    set(gca,'fontsize',15,'fontname','times')
    legend('$x(t)$','$y(t)$','Interpreter','latex')
    nexttile
    plot(yTrain(1,:),yTrain(2,:))
    set(gca,'fontsize',15,'fontname','times')
    xlabel('$x$','Interpreter','latex')
    ylabel('$y$','Interpreter','latex')
end

function plotSolutions(loss,yTrain,yPred)
    tiledlayout(1,2);
    nexttile
    plot(log10(loss))
    set(gca,'fontsize',15,'fontname','times')
    xlabel('iterazione','Interpreter','latex')
    ylabel('$\log_{10}(\mathcal{L}(\theta))$','Interpreter','latex')
    nexttile
    plot(yTrain(1,:),yTrain(2,:),"b-")
    hold on
    plot(yPred(1,:),yPred(2,:),"r--")
    hold off
    set(gca,'fontsize',15,'fontname','times')
    xlabel('$x$','Interpreter','latex')
    ylabel('$y$','Interpreter','latex')
    legend('$\phi(t)$','$\phi_\theta(t)$','Interpreter','latex')
end

function [gridData,randomData]=evaluations(q1,q2,int1,int2)
    interval1=(0:int1/(q1-1):int1);
    interval2=(0:int2/(q2-1):int2);
    for i=1:q1
        for j=1:q2
            gridData((i-1)*q2+j,:)=[interval1(i) interval2(j)];
        end
    end
    for i=1:q1*q2
        randomData(i,:)=[int1*rand(1) int2*rand(1)];
    end
end

function [NTrain,tau,miniBatch,NNeuron,NLayer,learnRate]=evaluation(NTrainInt,tauInt,miniBatchInt,NNeuronInt,NLayerInt,learnRateInt)
    NTrain=round((NTrainInt(2)-NTrainInt(1))*rand(1)+NTrainInt(1));
    tau=(tauInt(2)-tauInt(1))*rand(1)+tauInt(1);
    miniBatch=round((miniBatchInt(2)-miniBatchInt(1))*rand(1)+miniBatchInt(1));
    NNeuron=round((NNeuronInt(2)-NNeuronInt(1))*rand(1)+NNeuronInt(1));
    NLayer=round((NLayerInt(2)-NLayerInt(1))*rand(1)+NLayerInt(1));
    learnRate=(learnRateInt(2)-learnRateInt(1))*rand(1)+learnRateInt(1);
end

function scatterData(q1,q2,int1,int2)
[dataGrid,dataRandom]=evaluations(q1,q2,int1,int2);
tiledlayout(1,2);
nexttile
scatter(dataGrid(:,1),dataGrid(:,2),"filled");
xlabel('$\lambda_1$','Interpreter','latex')
ylabel('$\lambda_2$','Interpreter','latex')
nexttile
scatter(dataRandom(:,1),dataRandom(:,2),"filled")
xlabel('$\lambda_1$','Interpreter','latex')
ylabel('$\lambda_2$','Interpreter','latex')
end

function scatterLoss(x,y,xLabel,yLabel)
scatter(x,log10(y))
xlabel(xLabel,'Interpreter','latex')
if nargin < 4
    yLabel='$\log_{10}(\mathcal{L}(\theta))$';
end
ylabel(yLabel,'Interpreter','latex')
end

function plotTrainings(pts,values,tags)
plot(pts,log10(values))
set(gca,'fontsize',15,'fontname','times')
xlabel('iterazione','Interpreter','latex')
ylabel('$\log_{10}(\mathcal{L}(\theta))$','Interpreter','latex')
legend(tags)
end

%% 3.2.1 Figura 3

scatterData(5,5,3,2)

%% 3.3.1 Figura 4

N_true=10000;
[~,t_true,yTrain,~]=lotkaVolterraModel(N_true);
plotModel(t_true,yTrain)

%% 3.3.2 Figura 5

[loss,yTrain,yPred]=simulateNODE(1000,0.1,200,200,20,5,0.05);
plotSolutions(loss,yTrain,yPred);

%% 4.1 Formula 4.2

Q=1000;
NIter=200;
loss=zeros([1 Q]);
params=zeros([Q 6]);
for i=1:Q
    [NTrain,tau,miniBatch,NNeuron,NLayer,learnRate]=evaluation([1000 10000],[0.01 1],[10 100],[10 50],[1 20],[0.01 0.5]);
    params(i,:)=[NTrain,tau,miniBatch,NNeuron,NLayer,learnRate];
    [lossValues,~,~]=simulateNODE(NTrain,tau,miniBatch,NIter,NNeuron,NLayer,learnRate);
    loss(i)=lossValues(NIter);
end

data=zeros([Q 7]);
for i=1:Q
    data(i,1:6)=params(i,:);
    data(i,7)=loss(i);
end

%% 4.1 Figura 6

[loss,yTrain,yPred]=simulateNODE(4023,0.0237,35,1000,23,3,0.0177);
plotSolutions(loss(1:200),yTrain,yPred);

%% 4.2 Figura 7

tiledlayout(3,2)
nexttile
scatterLoss(data(:,5),data(:,7),'$L$')
nexttile
scatterLoss(data(:,4),data(:,7),'$n_\ell$')
nexttile
scatterLoss(data(:,1),data(:,7),'$m$')
nexttile
scatterLoss(data(:,3),data(:,7),'$m''$')
nexttile
scatterLoss(data(:,6),data(:,7),'$\alpha$')
nexttile
scatterLoss(data(:,2),data(:,7),'$\tau$')

%% 4.2.1 Figura 8

tags={};
i=0;
for num=2:2:10
    i=i+1;
    [loss(i,:),~,~] = simulateNODE(4023,0.0237,35,2000,23,num,0.0177);
    tags=[tags num2str(num)];
end

values=[];
pts=(0:10:1000);
for i=1:101
    values(i,:)=loss(:,i);
end
plotTrainings(pts,values,tags)

%% 4.2.1 Figura 9

tags={};
i=0;
for num=20:20:100
    i=i+1;
    [loss(i,:),~,~] = simulateNODE(4023,0.0237,35,1000,num,3,0.0177);
    tags=[tags num2str(num)];
end

values=[];
pts=(0:10:1000);
for i=1:101
    values(i,:)=loss(:,i);
end
plotTrainings(pts,values,tags)

%% 4.2.2 Figura 10

tags={};
i=0;
for num=2000:2000:10000
    i=i+1;
    [loss(i,:),~,~] = simulateNODE(num,0.0237,35,1000,23,3,0.0177);
    tags=[tags num2str(num)];
end

values=[];
pts=(0:10:1000);
for i=1:101
    values(i,:)=loss(:,i);
end
plotTrainings(pts,values,tags)

%% 4.2.2 Figura 11

tags={};
i=0;
for num=20:20:100
    i=i+1;
    [loss(i,:),~,~] = simulateNODE(4023,0.0237,num,1000,23,3,0.0177);
    tags=[tags num2str(num)];
end

values=[];
pts=(0:10:1000);
for i=1:101
    values(i,:)=loss(:,i);
end
plotTrainings(pts,values,tags)

%% 4.2.2 Figura 12

tags={};
i=0;
tm=zeros([1 5]);
for num=20:20:100
    i=i+1;
    tic
    [loss(i,:),~,~] = simulateNODE(4023,0.0237,num,1000,23,3,0.0177);
    tm(i)=toc;
end

plot((20:20:100),tm)
set(gca,'fontsize',15,'fontname','times')
xlabel('$m''$','Interpreter','latex')
ylabel('tempo medio per iterazione [ms]','Interpreter','latex')

%% 4.2.3 Figura 13

tags={};
i=0;
for num=0.1:0.1:0.5
    i=i+1;
    [loss(i,:),~,~] = simulateNODE(4023,0.0237,35,1000,23,3,num);
    tags=[tags num2str(num)];
end

values=[];
pts=(0:10:1000);
for i=1:101
    values(i,:)=loss(:,i);
end
plotTrainings(pts,values,tags)

%% 4.2.3 Figura 14

tags={};
i=0;
for num=0.01:0.01:0.05
    i=i+1;
    [loss(i,:),~,~] = simulateNODE(4023,0.0237,35,1000,23,3,num);
    tags=[tags num2str(num)];
end

values=[];
pts=(0:10:1000);
for i=1:101
    values(i,:)=loss(:,i);
end
plotTrainings(pts,values,tags)

%% 4.2.4 Figura 15

tags={};
i=0;
for num=0.2:0.2:1
    i=i+1;
    [loss(i,:),~,~] = simulateNODE(4023,num,35,1000,23,3,0.0177);
    tags=[tags num2str(num)];
end

values=[];
pts=(0:10:1000);
for i=1:101
    values(i,:)=loss(:,i);
end
plotTrainings(pts,values,tags)

%% 4.2.4 Figura 16

tags={};
i=0;
for num=0.02:0.02:0.1
    i=i+1;
    [loss(i,:),~,~] = simulateNODE(4023,num,35,1000,23,3,0.0177);
    tags=[tags num2str(num)];
end

values=[];
pts=(0:10:1000);
for i=1:101
    values(i,:)=loss(:,i);
end
plotTrainings(pts,values,tags)

%% 4.2.4 Figura 17

i=0;
loss=[];
vals=[];
for num=0.02:0.02:1
    i=i+1;
    vals(i)=num;
    [loss(i),~,~] = simulateNODE2(4023,num,35,200,23,3,0.0177);
end

plot(vals,log10(loss))
set(gca,'fontsize',15,'fontname','times')
xlabel('$\tau$','Interpreter','latex')
ylabel('$\log_{10}(\mathcal{L}(\theta))$','Interpreter','latex')
