function W=rotY(a,F,o)

R = [cos(a),0,sin(a); 0,1,0;  -sin(a),0,cos(a)];
W = F-o;
W=R*W';
W= W'+o;

end
