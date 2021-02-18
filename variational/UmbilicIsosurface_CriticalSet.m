load('./Data/UmbilicIsosurface_ValsLocusChart.mat')
f=isosurface(X,Y,Z,Val,0);

%% save
verts = f.vertices;
faces = f.faces;
save('./Data/UmbilicIsosurface_isodata.mat','verts','faces')

%% plot
fig=figure();
p=patch(f);
view(3)
p.FaceColor='g';
hold on;
u=plot3(UmbilicData(1),UmbilicData(2),UmbilicData(3),'k*');
u.LineWidth=5;
u.MarkerSize=40;

xlabel('x_0')
ylabel('x_1')
zlabel('x_2')

a = zeros(1,3); % Save how much figure is rotated (rad)

%% Cusp line
load('./Data/CuspLine_Umbilics.mat','cuspsP')
as = axis;
lP = plot3(cuspsP(:,1),cuspsP(:,2),cuspsP(:,3),'-');
lP.LineWidth=5;
lP.Color=[0.9290    0.6940    0.1250];
axis(as);

%% Set title
title('Hyperbolic Umbilic Critical Set')

