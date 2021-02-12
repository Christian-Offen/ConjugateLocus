load('./Data/ValLocusChart.mat','x0','x1','x2')
load('./Data/LocationUmbilics.mat')
load('./Data/isodata.mat','faces')
load('./Data/LocusVerts.mat')

[X0,X1,X2]=meshgrid(x0,x1,x2);

g.vertices = LocusVerts;
g.faces = faces;


%% plot
figL=figure();
axis([-0.78,0.64,-0.46,0.38,-0.63,0.48]);
P=patch(g);
view(3)
P.FaceColor='g';
P.FaceAlpha=0.1;
P.EdgeAlpha=0.1;
hold on;

%% plot markers at bifurcation points and line of cusps
u = gobjects(1,4);
for j=1:4
    u(j)=plot3(UmbilicLocation(j,1),UmbilicLocation(j,2),UmbilicLocation(j,3),'k*');
    u(j).LineWidth=5;
    u(j).MarkerSize=40;
end

csps = gobjects(1,2);
load('./Data/CuspLine_Circle.mat','cusps')
csps(1)=plot3(cusps(2:end,1),cusps(2:end,2),cusps(2:end,3));
csps(1).LineWidth=5;
load('./Data/CuspLine_Umbilics.mat','cusps')
csps(2)=plot3(cusps(2:end,1),cusps(2:end,2),cusps(2:end,3));
csps(2).LineWidth=5;



%% zoom to umbilic singularity
%{
box = 0.3*[-1,1,-1,1,-1,1];
j=2;
azoom = [UmbilicLocation(j,1)+box(1:2),UmbilicLocation(j,2)+box(3:4),UmbilicLocation(j,3)+box(5:6)];
axis(azoom)
%}

%% Set title
title('Conjugate Locus')
xlabel('x_0')
ylabel('x_1')
zlabel('x_2')

