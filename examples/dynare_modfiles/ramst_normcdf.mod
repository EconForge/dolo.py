// Tests the normcdf() function, in the static M-file, and in a dynamic C file

var c k t;
varexo x;

parameters alph gam delt bet aa;
alph=0.5;
gam=0.5;
delt=0.02;
bet=0.05;
aa=0.5;


model(use_dll);
c + k - aa*x*k(-1)^alph - (1-delt)*k(-1);
c^(-gam) - (1+bet)^(-1)*(aa*alph*x(+1)*k^(alph-1) + 1 - delt)*c(+1)^(-gam);
t = normcdf(x, 2, 3);
end;

initval;
x = 1;
k = ((delt+bet)/(1.0*aa*alph))^(1/(alph-1));
c = aa*k^alph-delt*k;
t = 0;
end;

steady;

check;

shocks;
var x;
periods 1;
values 1.2;
end;

simul(periods=20);

if (abs(oo_.steady_state(3) - normcdf(1, 2, 3)) > 1e-10)
   error('Test failed in static M-file')
end

if (abs(oo_.endo_simul(3, 2) - normcdf(1.2, 2, 3)) > 1e-10)
   error('Test failed in dynamic C file')
end
