// Example of optimal simple rule

var y inflation r;
varexo y_ inf_;

parameters delta sigma alpha kappa gammarr gammax0 gammac0 gamma_y_ gamma_inf_;

delta =  0.44;
kappa =  0.18;
alpha =  0.48;
sigma = -0.06;


model(linear);
y  = delta * y(-1)  + (1-delta)*y(+1)+sigma *(r - inflation(+1)) + y_; 
inflation  =   alpha * inflation(-1) + (1-alpha) * inflation(+1) + kappa*y + inf_;
r = gammax0*y(-1)+gammac0*inflation(-1)+gamma_y_*y_+gamma_inf_*inf_;
end;

shocks;
var y_;
stderr 0.63;
var inf_;
stderr 0.4;
end;


optim_weights;
inflation 1;
y 1;
end;

osr_params gammax0 gammac0 gamma_y_ gamma_inf_;

gammarr = 0;
gammax0 = 0.2;
gammac0 = 1.5;
gamma_y_ = 8;
gamma_inf_ = 3;

osr;
