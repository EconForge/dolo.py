var y r w c eh;

varexo e_y e_r;

parameters beta gamma rho_y rho_r ybar rbar theta;
beta = 0.96;
gamma = 4.0;
rho_y = 0.5;
rho_r = 0.5;
rbar = 1.0;
ybar = 1.0;
theta = 0.5;

model;

[eq_type='f' , complementarity='0 <= c <= w + y']

beta*eh*(w-c)^(theta-1)*theta / c^(-gamma) = 1 ;

[eq_type='g']
y = ybar + rho_y*(y(-1)-ybar) + e_y;

[eq_type='g']
r = rbar + rho_r*(r(-1)-rbar) + e_r;

[eq_type='g']
w = (w(-1)-c(-1))^(theta)*(rbar + rho_r*(r(-1)-rbar) + e_r) + (ybar + rho_y*(y(-1)-ybar) + e_y);

[eq_type='h']
eh = c(1)^(-gamma)*r(1);

end;


initval;
r = rbar;
y = ybar;
w = (    (1/(beta*theta*rbar))^(1/(theta-1))   )^theta*r+y;
c = w - (1/(beta*theta*rbar))^(1/(theta-1));
eh = c^(-gamma)*r;

end;
