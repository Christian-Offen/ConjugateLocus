load('./Data/LocationUmbilics.mat')
load('./Data/ValLocusChart.mat')

%% cut off data coming from singularities of charts

% set of interest lies within two ellipsoids

bnd = [-0.2115    0.4037   -0.3251    0.4164 -0.2094    0.5237];

d0 =(bnd(2)-bnd(1))*1.0;
d1 =(bnd(4)-bnd(3))*1.0;
d2 =(bnd(6)-bnd(5))*1.0;
r=[d0,d1,d2]/2;
ri = 2*r/3;

x0c =(bnd(2)+bnd(1))/2;
x1c =(bnd(4)+bnd(3))/2;
x2c =(bnd(6)+bnd(5))/2;
xc = [x0c,x1c,x2c];

%[xe,ye,ze]=ellipsoid(x0c,x1c,x2c,d0/2,d1/2,d2/2);
%S=surf(xe,ye,ze);
%S.FaceAlpha=0.3;
%S.EdgeAlpha=0.3;

[X0,X1,X2]=meshgrid(x0,x1,x2);

% outer ellipsoid
excl = (X0-xc(1)).^2/r(1).^2 + (X1-xc(2)).^2/r(2).^2+(X2-xc(3)).^2/r(3).^2>1;
X0(excl)=NaN;
X1(excl)=NaN;
X2(excl)=NaN;

% inner ellipsoid
excl = (X0-xc(1)).^2/ri(1).^2 + (X1-xc(2)).^2/ri(2).^2+(X2-xc(3)).^2/ri(3).^2<1;
X0(excl)=NaN;
X1(excl)=NaN;
X2(excl)=NaN;

%%

f=isosurface(X0,X1,X2,Val,0);

%% save
verts = f.vertices;
faces = f.faces;
save('Data/isodata.mat','verts','faces')

%% plot
fig=figure();
p=patch(f);
view(3)
p.FaceColor='g';
p.EdgeAlpha=0.1;
p.FaceAlpha=0.1;
hold on;



%% plot markers at bifurcation points and line of cusps
u = gobjects(1,4);
for j=1:4
    u(j)=plot3(UmbilicLocationPreimage(j,1),UmbilicLocationPreimage(j,2),UmbilicLocationPreimage(j,3),'k*');
    u(j).LineWidth=5;
    u(j).MarkerSize=40;
end

csps = gobjects(1,2);
load('./Data/CuspLine_Umbilics.mat','cuspsP')
csps(1)=plot3(cuspsP(2:end,1),cuspsP(2:end,2),cuspsP(2:end,3));
csps(1).LineWidth=5;
load('./Data/CuspLine_Circle.mat','cuspsP')
csps(2)=plot3(cuspsP(2:end,1),cuspsP(2:end,2),cuspsP(2:end,3));
csps(2).LineWidth=5;


%% Set title
title('Critical Set')

xlabel('x_0')
ylabel('x_1')
zlabel('x_2')


%%
% fig.Renderer='Painter';
% orient landscape
% print(fig,'preimage.pdf','-dpdf','-fillpage')