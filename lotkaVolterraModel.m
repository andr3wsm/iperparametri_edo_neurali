function [trueModel,t_true,yTrain,y0]=lotkaVolterraModel(N_true)

y0=[6;2];
T=10;
t_true=linspace(0,T,N_true);
a=1.5;
b=1.0;
c=3.0;
d=1.0;
trueModel=@(t,y)[a*y(1)-b*y(1)*y(2); -c*y(2)+d*y(1)*y(2)];

odeOptions=odeset(RelTol=1.e-7);
[~,yTrain]=ode45(trueModel,t_true,y0,odeOptions);
yTrain=yTrain';

end