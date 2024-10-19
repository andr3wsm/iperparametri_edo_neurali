%% 3.3.1 Figura 3

N_true=10000;
[~,t_true,yTrain,~]=lotkaVolterraModel(N_true);

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