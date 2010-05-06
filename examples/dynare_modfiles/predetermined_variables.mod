var c k i;
varexo x;

parameters alph gam delt bet aa;
alph=0.5;
gam=0.5;
delt=0.02;
bet=0.05;
aa=0.5;

predetermined_variables k;

model;
c + i = aa*x*k^alph;
c^(-gam) - (1+bet)^(-1)*(aa*alph*x(+1)*k(+1)^(alph-1) + 1 - delt)*c(+1)^(-gam);
k(+1) = (1-delt)*k + i;
end;

initval;
x = 1;
k = ((delt+bet)/(1.0*aa*alph))^(1/(alph-1));
c = aa*k^alph-delt*k;
i = delt*k;
end;

write_latex_dynamic_model;

steady;

check;

shocks;
var x;
periods 1;
values 1.2;
end;

simul(periods=200);

rplot c;
rplot k;
