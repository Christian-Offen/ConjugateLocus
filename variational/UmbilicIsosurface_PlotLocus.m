load('./Data/UmbilicIsosurface_isodata.mat','faces')
load('./Data/UmbilicIsosurface_LocusVerts.mat')
load('./Data/CuspLine_Umbilics.mat','cusps')

g.vertices = LocusVerts;
g.faces = faces;


%% plot
figL=figure();
P=patch(g);
view(3)
P.FaceColor='g';
P.FaceAlpha=0.2;
P.EdgeAlpha=0.5;
hold on;
u=plot3(LocusUmbilic(1),LocusUmbilic(2),LocusUmbilic(3),'k*');
u.LineWidth=5; 
u.MarkerSize=40;

% plot line of cusps
as = axis;
l=plot3(cusps(:,1),cusps(:,2),cusps(:,3),'-');
l.LineWidth=5;
axis(as);

xlabel('x_0')
ylabel('x_1')
zlabel('x_2')

a = zeros(1,3); % Save how much figure is rotated (rad)

%% Rotation X0

da(1) = -0.0540;
a(1) = a(1)+da(1);

P.Vertices = rotX(da(1),P.Vertices,LocusUmbilic);
Cusps = rotX(da(1),[l.XData',l.YData',l.ZData'],LocusUmbilic);
l.XData=Cusps(:,1);
l.YData=Cusps(:,2);
l.ZData=Cusps(:,3);

%% Rotation X1

da(2) = 1.0200;
a(2) = a(2)+da(2);

P.Vertices = rotY(da(2),P.Vertices,LocusUmbilic);
Cusps = rotY(da(2),Cusps,LocusUmbilic);
l.XData=Cusps(:,1);
l.YData=Cusps(:,2);
l.ZData=Cusps(:,3);

%% Rotation X2

%da(3) = 0.0;
%a(3) = a(3)+da(3);
%P.Vertices = rotZ(da(3),P.Vertices,LocusUmbilic);

%% Set title
title({'Hyperbolic Umbilic',['rotated around x_0-direction by ' mat2str(a(1)) 'rad then x_1-direction by ' mat2str(a(2)) 'rad']})

axis([-0.4128   -0.4120   -0.0415   -0.0397   -0.3800   -0.3300]);