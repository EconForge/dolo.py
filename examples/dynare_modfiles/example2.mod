// Example 2 from Collard's guide to Dynare
var y, c, k, a, h, b;
varexo e, u;

parameters beta, rho, alpha, delta, theta, psi, tau;

alpha = 0.36;
rho   = 0.95;
tau   = 0.025;
beta  = 0.99;
delta = 0.025;
psi   = 0;
theta = 2.95;

model;
exp(c)*theta*exp(h)^(1+psi)=(1-alpha)*exp(y);
exp(k) = beta*(((exp(b)*exp(c))/(exp(b(+1))*exp(c(+1))))
         *(exp(b(+1))*alpha*exp(y(+1))+(1-delta)*exp(k)));
exp(y) = exp(a)*(exp(k(-1))^alpha)*(exp(h)^(1-alpha));
exp(k) = exp(b)*(exp(y)-exp(c))+(1-delta)*exp(k(-1));
a = rho*a(-1)+tau*b(-1) + e;
b = tau*a(-1)+rho*b(-1) + u;
end;

initval;
y = 0.1;
c = -0.2;
h = -1.2;
k =  2.4;
a = 0;
b = 0;
e = 0;
u = 0;
end;

steady;

shocks;
var e = 0.009^2;
var u = 0.009^2;
end;

stoch_simul(periods=2000, drop=200);
