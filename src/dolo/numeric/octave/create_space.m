function [interp] = create_space(tt, order, smin, smax)

global interp;

interp = struct;
interp.fspace = fundefn(tt,order,smin,smax);       % function space
interp.Phi    = funbasx(interp.fspace);
interp.snodes = funnode(interp.fspace);                   % state collocation nodes
interp.points = gridmake(interp.snodes);    % the grid on which the function is known
