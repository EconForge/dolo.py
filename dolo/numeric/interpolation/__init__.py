def create_interpolator(grid, interp_method=None):
    if interp_method is None:
        interp_method = grid.interpolation
    if grid.__class__.__name__ == 'SmolyakGrid':
        if interp_method is not None and interp_method not in ('chebychev','polynomial'):
            raise Exception("Interpolation method '{}' is not implemented for Smolyak grids.".format(interp_method))
        from dolo.numeric.interpolation.smolyak import SmolyakGrid
        dr = SmolyakGrid(grid.a, grid.b, grid.mu)
    elif grid.__class__.__name__ == 'CartesianGrid':
        if interp_method is not None and interp_method not in ('spline', 'cspline',' splines','csplines'):
            raise Exception("Interpolation method '{}' is not implemented for Cartesian grids.".format(interp_method))
        from dolo.numeric.interpolation.splines import MultivariateSplines
        dr = MultivariateSplines(grid.a, grid.b, grid.orders)
    else:
        raise Exception("Unknown grid type '{}'.".format(grid))
    return dr
