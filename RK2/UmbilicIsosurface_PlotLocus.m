load('./Data/UmbilicIsosurface_isodata.mat','faces')
load('./Data/UmbilicIsosurface_LocusVerts.mat')

g.vertices = LocusVerts;
g.faces = faces;


%% plot
figL=figure();
P=patch(g);
view(3)
P.FaceColor='g';
hold on;
u=plot3(LocusUmbilic(1),LocusUmbilic(2),LocusUmbilic(3),'k*');
u.LineWidth=5;
u.MarkerSize=40;

xlabel('x_0')
ylabel('x_1')
zlabel('x_2')

a = zeros(1,3); % Save how much figure is rotated (rad)

%% Rotation X0

da(1) = -0.075;
a(1) = a(1)+da(1);

P.Vertices = rotX(da(1),P.Vertices,LocusUmbilic);

%% Rotation X1

da(2) = 1.12;
a(2) = a(2)+da(2);

P.Vertices = rotY(da(2),P.Vertices,LocusUmbilic);

%% Rotation X2

da(3) = 0.0;
a(3) = a(3)+da(3);

P.Vertices = rotZ(da(3),P.Vertices,LocusUmbilic);

%% Set title
title({'Hyperbolic Umbilic',['rotated around x_0-direction by ' mat2str(a(1)) 'rad then x_1-direction by ' mat2str(a(2)) 'rad']})

%%
% figL.Renderer='Painter';
% orient landscape
% print(figL,'HyperbolicUmbilic.pdf','-dpdf','-fillpage')