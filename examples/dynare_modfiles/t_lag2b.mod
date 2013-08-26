// example 1 from Collard's guide to Dynare
periods 400;

var y, c, k, a, h, b, b1;
varexo e,u;

parameters beta, rho, alpha, delta, theta, psi, tau, phi;

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
k = beta*(((exp(b)*c)/(exp(b(+1))*c(+1)))
    *(exp(b(+1))*alpha*y(+1)+(1-delta)*k));
y = exp(a)*(k(-1)^alpha)*(h^(1-alpha));
k = exp(b)*(y-c)+(1-delta)*k(-1);
a = rho*a(-1)+tau*b1(-1) + e;
b = tau*a(-1)+rho*b1(-1) + u;
b1 = b(-1);
end;

initval;
y = 1.08068253095672;
c = 0.80359242014163;
h = 0.29175631001732;
k = 5;
a = 0;
b = 0;
b1 = 0;
end;

Sigma_e = [ 0.000081; (phi*0.009*0.009) 0.000081];

stoch_simul(irf=0,periods=10000,order=2);
