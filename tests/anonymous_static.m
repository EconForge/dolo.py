function [residual, g1, g2] = anonymous_static(y, x, params, it_)
%
% Status : Computes static model for Dynare
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

%
% Model equations
%

    residual = zeros(6, 1);
    residual(1) = -(1 - params(3))*y(1) + y(5)^(1 + params(6))*params(5)*y(2);
    residual(2) = -((1 - params(4))*y(3) + params(3)*y(1)*exp(y(6)))*params(1) + y(3);
    residual(3) = y(1) - y(5)^(1 - params(3))*y(3)^params(3)*exp(y(4));
    residual(4) = -(1 - params(4))*y(3) + y(3) - (-y(2) + y(1))*exp(y(6));
    residual(5) = -x(1) - params(2)*y(4) - params(7)*y(6) + y(4);
    residual(6) = -x(2) - params(2)*y(6) - params(7)*y(4) + y(6);

    if nargout >= 2
        g1 = zeros(6, 6);

%
% Jacobian matrix
%

        g1(1,1) = -1 + params(3);
        g1(1,2) = y(5)^(1 + params(6))*params(5);
        g1(1,5) = y(5)^(1 + params(6))*(1 + params(6))*params(5)*y(2)/y(5);
        g1(2,1) = -params(3)*params(1)*exp(y(6));
        g1(2,3) = 1 - (1 - params(4))*params(1);
        g1(2,6) = -params(3)*params(1)*y(1)*exp(y(6));
        g1(3,3) = -y(5)^(1 - params(3))*y(3)^params(3)*params(3)*exp(y(4))/y(3);
        g1(3,1) = 1;
        g1(3,4) = -y(5)^(1 - params(3))*y(3)^params(3)*exp(y(4));
        g1(3,5) = -y(5)^(1 - params(3))*y(3)^params(3)*(1 - params(3))*exp(y(4))/y(5);
        g1(4,1) = -exp(y(6));
        g1(4,3) = params(4);
        g1(4,2) = exp(y(6));
        g1(4,6) = -(-y(2) + y(1))*exp(y(6));
        g1(5,6) = -params(7);
        g1(5,4) = 1 - params(2);
        g1(6,6) = 1 - params(2);
        g1(6,4) = -params(7);

    end

    if nargout >= 3
        g2 = zeros(6, 36);

%
% Hessian matrix
%

        g2(1,26) = y(5)^(1 + params(6))*(1 + params(6))*params(5)/y(5);
        g2(1,11) = g2(1,26);
        g2(1,29) = (1 + params(6))^2*y(5)^(1 + params(6))*params(5)*y(2)/y(5)^2 - y(5)^(1 + params(6))*(1 + params(6))*params(5)*y(2)/y(5)^2;
        g2(2,31) = -params(3)*params(1)*exp(y(6));
        g2(2,6) = g2(2,31);
        g2(2,36) = -params(3)*params(1)*y(1)*exp(y(6));
        g2(3,15) = y(5)^(1 - params(3))*y(3)^params(3)*params(3)*exp(y(4))/y(3)^2 - params(3)^2*y(5)^(1 - params(3))*y(3)^params(3)*exp(y(4))/y(3)^2;
        g2(3,21) = -y(5)^(1 - params(3))*y(3)^params(3)*params(3)*exp(y(4))/y(3);
        g2(3,16) = g2(3,21);
        g2(3,27) = -y(5)^(1 - params(3))*y(3)^params(3)*(1 - params(3))*params(3)*exp(y(4))/(y(5)*y(3));
        g2(3,17) = g2(3,27);
        g2(3,22) = -y(5)^(1 - params(3))*y(3)^params(3)*exp(y(4));
        g2(3,28) = -y(5)^(1 - params(3))*y(3)^params(3)*(1 - params(3))*exp(y(4))/y(5);
        g2(3,23) = g2(3,28);
        g2(3,29) = y(5)^(1 - params(3))*y(3)^params(3)*(1 - params(3))*exp(y(4))/y(5)^2 - (1 - params(3))^2*y(5)^(1 - params(3))*y(3)^params(3)*exp(y(4))/y(5)^2;
        g2(4,31) = -exp(y(6));
        g2(4,6) = g2(4,31);
        g2(4,32) = exp(y(6));
        g2(4,12) = g2(4,32);
        g2(4,36) = -(-y(2) + y(1))*exp(y(6));

    end
