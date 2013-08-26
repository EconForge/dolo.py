// Test USE_DLL option at order 2

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

phi   = 0.1;

model(use_dll);
c*theta*h^(1+psi)=(1-alpha)*y;
k = beta*(((exp(b)*c)/(exp(b(+1))*c(+1)))
    *(exp(b(+1))*alpha*y(+1)+(1-delta)*k));
y = exp(a)*(k(-1)^alpha)*(h^(1-alpha));
k = exp(b)*(y-c)+(1-delta)*k(-1);
a = rho*a(-1)+tau*b(-1) + e;
b = tau*a(-1)+rho*b(-1) + u;
end;

initval;
y = 1.08068253095672;
c = 0.80359242014163;
h = 0.29175631001732;
k = 11.08360443260358;
a = 0;
b = 0;
e = 0;
u = 0;
end;

shocks;
var e; stderr 0.009;
var u; stderr 0.009;
var e, u = phi*0.009*0.009;
end;

stoch_simul(nograph);

if ~exist('example1_results.mat','file');
   error('example1 must be run first');
end;

oo1 = load('example1_results','oo_');

dr0 = oo1.oo_.dr;
dr = oo_.dr;

if max(max(abs(dr0.ghx - dr.ghx))) > 1e-12;
   error('error in ghx');
end;
if max(max(abs(dr0.ghu - dr.ghu))) > 1e-12;
   error('error in ghu');
end;
if max(max(abs(dr0.ghxx - dr.ghxx))) > 1e-12;
   error('error in ghxx');
end;
if max(max(abs(dr0.ghuu - dr.ghuu))) > 1e-12;
   error('error in ghuu');
end;
if max(max(abs(dr0.ghxu - dr.ghxu))) > 1e-12;
   error('error in ghxu');
end;
if max(max(abs(dr0.ghs2 - dr.ghs2))) > 1e-12;
   error('error in ghs2');
end;
