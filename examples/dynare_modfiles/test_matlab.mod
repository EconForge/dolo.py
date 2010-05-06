periods 200;
var y, c, k, a, h, b;
varexo e,u;

parameters beta, rho, beta, alpha, delta, theta, psi, tau, phi;

alpha = 0.36;
rho   = 0.95;
tau   = 0.025;
beta  = 0.99;
delta = 0.025;
psi   = 0;
theta = 2.95;

phi   = 0.1;

model;
c*theta*h^(1+psi)=(1-alpha)*y;
k = beta*(((exp(b)*c)/(exp(b(+1))*c(+1)))*(exp(b(+1))*alpha*y(+1)+(1-delta)*k));
y = exp(a)*(k(-1)^alpha)*(h^(1-alpha));
k = exp(b)*(y-c)+(1-delta)*k(-1);
a = rho*a(-1)+tau*b(-1) + e;
b = tau*a(-1)+rho*b(-1) + u;
end;

initval;
y = 1;
c = 0.7;
h = 0.1;
k = 11;
a = 0;
b = 0;
e = 0;
u = 0;
end;

Sigma_e = [ 0.01 0.005; 0.01];

stoch_simul(irf=0);

