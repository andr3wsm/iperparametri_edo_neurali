% output:
%         valori della funzione di costo
%         |          dati di addestramento
%         |          |      valori predetti
function [trueLoss,yTrain,yPred]=simulateNODE2(N_true,tau,miniBatchSize,N_iter,N_neuron,N_hiddenLayer,learnRate)

N_layer=1+N_hiddenLayer;

%% modello reale e dati di addestramento
[~,t_true,yTrain,y0]=lotkaVolterraModel(N_true);
N_train=N_true;

%% passi dell'e.d.o. neurale
h_node=t_true(2);           % dimensione del passo
t_node=(0:h_node:tau);   % valori di campionamento
N_node=size(t_node);
N_node=N_node(2)-1;

%% parametri di Adam
gradDecay=0.9;
sqGradDecay=0.999;
averageGrad=[];
averageSqGrad=[];

%% definizione dell'e.d.o. neurale
NODE=struct;
stateSize=size(yTrain,1);
hiddenSize=N_neuron;
NODE.depth=dlarray(N_layer);
for l=1:NODE.depth
    % determinazione delle dimensioni
    sizeIn=hiddenSize;
    sizeOut=hiddenSize;
    if l==1
        sizeIn=stateSize;
    elseif l==NODE.depth
        sizeOut=stateSize;
    end
    % inizializzazione del livello
    NODE.(layerName(l))=struct;
    NODE.(layerName(l)).Weights=initializeGlorot(sizeIn,sizeOut);
    NODE.(layerName(l)).Bias=initializeZero(1,sizeOut);
end

%% addestramento
iter=0;
lossValues=zeros(1,N_iter);
while iter<N_iter
    iter=iter+1;             % iterazione attuale
    % creazione del minibatch
    [Y,targets]=createMiniBatch(N_train,N_node,miniBatchSize,yTrain);
    % calcolo della funzione di costo
    [loss,gradients]=dlfeval(@modelLoss,t_node,Y,NODE,targets);
    % aggiornamento della rete con Adam
    [NODE,averageGrad,averageSqGrad]=adamupdate(NODE,gradients,...
        averageGrad,averageSqGrad,iter,...
        learnRate,gradDecay,sqGradDecay);
    lossValues(iter)=loss;
end

trueLoss = 0;
yPred=dlode45(@NODEModel,t_true,dlarray(y0),NODE,DataFormat="CB");
for i=1:N_true
    trueLoss=trueLoss + norm(extractdata(yTrain(i)-yPred(i)));
end
trueLoss=trueLoss/(2*N_true);

end

%% funzioni modello dell'e.d.o. neurale
% funzione di attivazione
function y=activate(z)
y=tanh(z);
end
% passo in avanti
function y=NODEModel(~,y,theta)
for l=1:theta.depth
    y=theta.(layerName(l)).Weights*y+theta.(layerName(l)).Bias;
    if l~=theta.depth
        y=activate(y);
    end
end
end
% nome dei livelli
function n=layerName(l)
    n=['fc' num2str(l)];
end

%% funzione di costo
function [loss,gradients]=modelLoss(tspan,y0,NODE,targets)
Y=dlode45(@NODEModel,tspan,y0,NODE,DataFormat="CB");
loss=l2loss(Y,targets,NormalizationFactor="all-elements",DataFormat="CBT");
gradients=dlgradient(loss,NODE);
end

%% creazione del minibatch
function [y0,targets]=createMiniBatch(numTimesteps,numTimesPerObs,miniBatchSize,Y)
s=randperm(numTimesteps - numTimesPerObs,miniBatchSize);
y0=dlarray(Y(:,s));
targets=zeros([size(Y,1) miniBatchSize numTimesPerObs]);
for i=1:miniBatchSize
    targets(:,i,1:numTimesPerObs)=Y(:,s(i)+1:(s(i)+numTimesPerObs));
end
end

%% funzioni di inizializzazione
function parameters=initializeGlorot(sizeIn, sizeOut)
Z=2*rand([sizeOut sizeIn],'single')-1;
bound=sqrt(6/(sizeIn+sizeOut));
parameters=dlarray(bound*Z);
end
function parameters=initializeZero(sizeIn, sizeOut)
parameters=dlarray(zeros([sizeOut sizeIn],'single'));
end