// Example of Ramsey policy computation

var y inflation r;
varexo y_ inf_;

parameters delta sigma alpha kappa gammarr gammax0 gammac0 rbar lambda1 lambda2;

delta =  0.44;
kappa =  0.18;
alpha =  0.48;
sigma = -0.06;
lambda1 = 0.5;
lambda2 = 0.1;

model(linear);
y  = delta * y(-1) + (1-delta) * y(+1) + sigma *(r - inflation(+1)) + y_; 
inflation  =   alpha * inflation(-1) + (1-alpha) * inflation(+1) + kappa*y + inf_;
end;

shocks;
var y_;
stderr 0.63;
var inf_;
stderr 0.4;
end;

planner_objective inflation^2 + lambda1*y^2 + lambda2*r^2;

ramsey_policy(planner_discount=0.95, order = 1);
