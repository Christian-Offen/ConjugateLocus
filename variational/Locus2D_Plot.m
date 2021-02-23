% Plot locus on ellipsoid

load('./Data/Locus2D.mat')

El=mesh(y,z,x);
El.EdgeColor = 'k';
El.EdgeAlpha=0.1;
El.FaceAlpha=0.1;

hold on
lc = plot3(lc(2,:),lc(3,:),lc(1,:));
lc.LineWidth=5;
lc.Color='k';


pt = plot3(ref(2),ref(3),ref(1));
pt.MarkerSize=20;
pt.Marker='*';
pt.Color='k';
pt.LineWidth=4;

xlabel('x_1')
ylabel('x_2')
zlabel('x_0')

set(gca,'fontsize',20)

%%
% orient(gcf,'landscape')
% set(gcf,'renderer','painters');
% print(gcf,'./Plots/2DLocus_Painter.pdf','-dpdf','-fillpage')