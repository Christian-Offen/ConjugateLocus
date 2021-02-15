function W=rotX(a,F,o)

R = [1,0,0; 0, cos(a),-sin(a); 0, sin(a),cos(a)];
W = F-o;
W=R*W';
W= W'+o;

end