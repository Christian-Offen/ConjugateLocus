function W=rotZ(a,F,o)

R = [cos(a),-sin(a),0; sin(a),cos(a),0;0,0,1];
W = F-o;
W=R*W';
W= W'+o;

end