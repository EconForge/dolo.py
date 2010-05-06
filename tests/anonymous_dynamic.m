function [residual, g1, g2] = anonymous_dynamic(y, x, params, it_)
%
% Status : Computes dynamic model for Dynare
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

%
% Model equations
%

    residual = zeros(6, 1);
    residual(1) = -(1 - params(3))*y(4) + y(8)^(1 + params(6))*params(5)*y(5);
    residual(2) = y(6) - ((1 - params(4))*y(6) + params(3)*y(10)*exp(y(12)))*params(1)*y(5)*exp(-y(12))*exp(y(9))/y(11);
    residual(3) = y(4) - y(8)^(1 - params(3))*y(1)^params(3)*exp(y(7));
    residual(4) = -(1 - params(4))*y(1) + y(6) - (-y(5) + y(4))*exp(y(9));
    residual(5) = -x(it_,1) - params(2)*y(2) - params(7)*y(3) + y(7);
    residual(6) = -x(it_,2) - params(2)*y(3) - params(7)*y(2) + y(9);

    if nargout >= 2
%
% Jacobian matrix
%

        v1 = zeros(26, 3);
        v1(1,:) = [1, 4, -1 + params(3)] ;
        v1(2,:) = [1, 5, y(8)^(1 + params(6))*params(5)] ;
        v1(3,:) = [1, 8, y(8)^(1 + params(6))*(1 + params(6))*params(5)*y(5)/y(8)] ;
        v1(4,:) = [2, 9, -((1 - params(4))*y(6) + params(3)*y(10)*exp(y(12)))*params(1)*y(5)*exp(-y(12))*exp(y(9))/y(11)] ;
        v1(5,:) = [2, 6, 1 - (1 - params(4))*params(1)*y(5)*exp(-y(12))*exp(y(9))/y(11)] ;
        v1(6,:) = [2, 11, ((1 - params(4))*y(6) + params(3)*y(10)*exp(y(12)))*params(1)*y(5)*exp(-y(12))*exp(y(9))/y(11)^2] ;
        v1(7,:) = [2, 5, -((1 - params(4))*y(6) + params(3)*y(10)*exp(y(12)))*params(1)*exp(-y(12))*exp(y(9))/y(11)] ;
        v1(8,:) = [2, 12, ((1 - params(4))*y(6) + params(3)*y(10)*exp(y(12)))*params(1)*y(5)*exp(-y(12))*exp(y(9))/y(11) - params(3)*params(1)*y(5)*y(10)*exp(y(9))/y(11)] ;
        v1(9,:) = [2, 10, -params(3)*params(1)*y(5)*exp(y(9))/y(11)] ;
        v1(10,:) = [3, 1, -y(8)^(1 - params(3))*y(1)^params(3)*params(3)*exp(y(7))/y(1)] ;
        v1(11,:) = [3, 4, 1] ;
        v1(12,:) = [3, 7, -y(8)^(1 - params(3))*y(1)^params(3)*exp(y(7))] ;
        v1(13,:) = [3, 8, -y(8)^(1 - params(3))*y(1)^params(3)*(1 - params(3))*exp(y(7))/y(8)] ;
        v1(14,:) = [4, 4, -exp(y(9))] ;
        v1(15,:) = [4, 6, 1] ;
        v1(16,:) = [4, 5, exp(y(9))] ;
        v1(17,:) = [4, 9, -(-y(5) + y(4))*exp(y(9))] ;
        v1(18,:) = [4, 1, -1 + params(4)] ;
        v1(19,:) = [5, 13, -1] ;
        v1(20,:) = [5, 3, -params(7)] ;
        v1(21,:) = [5, 2, -params(2)] ;
        v1(22,:) = [5, 7, 1] ;
        v1(23,:) = [6, 3, -params(2)] ;
        v1(24,:) = [6, 2, -params(7)] ;
        v1(25,:) = [6, 9, 1] ;
        v1(26,:) = [6, 14, -1] ;
        g1 = sparse(v1(:,1),v1(:,2),v1(:,3),6,14);

    end

    if nargout >= 3
%
% Hessian matrix
%

        v2 = zeros(48, 3);
        v2(1,:) = [1, 103, y(8)^(1 + params(6))*(1 + params(6))*params(5)/y(8)] ;
        v2(2,:) = [1, 64, v2(1,3)];
        v2(3,:) = [1, 106, (1 + params(6))^2*y(8)^(1 + params(6))*params(5)*y(5)/y(8)^2 - y(8)^(1 + params(6))*(1 + params(6))*params(5)*y(5)/y(8)^2] ;
        v2(4,:) = [2, 121, -((1 - params(4))*y(6) + params(3)*y(10)*exp(y(12)))*params(1)*y(5)*exp(-y(12))*exp(y(9))/y(11)] ;
        v2(5,:) = [2, 118, -(1 - params(4))*params(1)*y(5)*exp(-y(12))*exp(y(9))/y(11)] ;
        v2(6,:) = [2, 79, v2(5,3)];
        v2(7,:) = [2, 149, ((1 - params(4))*y(6) + params(3)*y(10)*exp(y(12)))*params(1)*y(5)*exp(-y(12))*exp(y(9))/y(11)^2] ;
        v2(8,:) = [2, 123, v2(7,3)];
        v2(9,:) = [2, 117, -((1 - params(4))*y(6) + params(3)*y(10)*exp(y(12)))*params(1)*exp(-y(12))*exp(y(9))/y(11)] ;
        v2(10,:) = [2, 65, v2(9,3)];
        v2(11,:) = [2, 163, ((1 - params(4))*y(6) + params(3)*y(10)*exp(y(12)))*params(1)*y(5)*exp(-y(12))*exp(y(9))/y(11) - params(3)*params(1)*y(5)*y(10)*exp(y(9))/y(11)] ;
        v2(12,:) = [2, 124, v2(11,3)];
        v2(13,:) = [2, 135, -params(3)*params(1)*y(5)*exp(y(9))/y(11)] ;
        v2(14,:) = [2, 122, v2(13,3)];
        v2(15,:) = [2, 146, (1 - params(4))*params(1)*y(5)*exp(-y(12))*exp(y(9))/y(11)^2] ;
        v2(16,:) = [2, 81, v2(15,3)];
        v2(17,:) = [2, 75, -(1 - params(4))*params(1)*exp(-y(12))*exp(y(9))/y(11)] ;
        v2(18,:) = [2, 62, v2(17,3)];
        v2(19,:) = [2, 160, (1 - params(4))*params(1)*y(5)*exp(-y(12))*exp(y(9))/y(11)] ;
        v2(20,:) = [2, 82, v2(19,3)];
        v2(21,:) = [2, 151, -2*((1 - params(4))*y(6) + params(3)*y(10)*exp(y(12)))*params(1)*y(5)*exp(-y(12))*exp(y(9))/y(11)^3] ;
        v2(22,:) = [2, 145, ((1 - params(4))*y(6) + params(3)*y(10)*exp(y(12)))*params(1)*exp(-y(12))*exp(y(9))/y(11)^2] ;
        v2(23,:) = [2, 67, v2(22,3)];
        v2(24,:) = [2, 165, params(3)*params(1)*y(5)*y(10)*exp(y(9))/y(11)^2 - ((1 - params(4))*y(6) + params(3)*y(10)*exp(y(12)))*params(1)*y(5)*exp(-y(12))*exp(y(9))/y(11)^2] ;
        v2(25,:) = [2, 152, v2(24,3)];
        v2(26,:) = [2, 150, params(3)*params(1)*y(5)*exp(y(9))/y(11)^2] ;
        v2(27,:) = [2, 137, v2(26,3)];
        v2(28,:) = [2, 159, ((1 - params(4))*y(6) + params(3)*y(10)*exp(y(12)))*params(1)*exp(-y(12))*exp(y(9))/y(11) - params(3)*params(1)*y(10)*exp(y(9))/y(11)] ;
        v2(29,:) = [2, 68, v2(28,3)];
        v2(30,:) = [2, 131, -params(3)*params(1)*exp(y(9))/y(11)] ;
        v2(31,:) = [2, 66, v2(30,3)];
        v2(32,:) = [2, 166, params(3)*params(1)*y(5)*y(10)*exp(y(9))/y(11) - ((1 - params(4))*y(6) + params(3)*y(10)*exp(y(12)))*params(1)*y(5)*exp(-y(12))*exp(y(9))/y(11)] ;
        v2(33,:) = [2, 164, 0] ;
        v2(34,:) = [2, 138, v2(33,3)];
        v2(35,:) = [3, 1, y(8)^(1 - params(3))*y(1)^params(3)*params(3)*exp(y(7))/y(1)^2 - params(3)^2*y(8)^(1 - params(3))*y(1)^params(3)*exp(y(7))/y(1)^2] ;
        v2(36,:) = [3, 85, -y(8)^(1 - params(3))*y(1)^params(3)*params(3)*exp(y(7))/y(1)] ;
        v2(37,:) = [3, 7, v2(36,3)];
        v2(38,:) = [3, 99, -y(8)^(1 - params(3))*y(1)^params(3)*(1 - params(3))*params(3)*exp(y(7))/(y(8)*y(1))] ;
        v2(39,:) = [3, 8, v2(38,3)];
        v2(40,:) = [3, 91, -y(8)^(1 - params(3))*y(1)^params(3)*exp(y(7))] ;
        v2(41,:) = [3, 105, -y(8)^(1 - params(3))*y(1)^params(3)*(1 - params(3))*exp(y(7))/y(8)] ;
        v2(42,:) = [3, 92, v2(41,3)];
        v2(43,:) = [3, 106, y(8)^(1 - params(3))*y(1)^params(3)*(1 - params(3))*exp(y(7))/y(8)^2 - (1 - params(3))^2*y(8)^(1 - params(3))*y(1)^params(3)*exp(y(7))/y(8)^2] ;
        v2(44,:) = [4, 116, -exp(y(9))] ;
        v2(45,:) = [4, 51, v2(44,3)];
        v2(46,:) = [4, 117, exp(y(9))] ;
        v2(47,:) = [4, 65, v2(46,3)];
        v2(48,:) = [4, 121, -(-y(5) + y(4))*exp(y(9))] ;
        g2 = sparse(v2(:,1),v2(:,2),v2(:,3),6,196);

    end
