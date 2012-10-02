from __future__ import division

import numpy
import numpy as np

from misc import cartesian

class RectangularDomain:
    def __init__(self,smin,smax,orders):
        self.d = len(smin)
        self.smin = smin
        self.smax = smax
        self.bounds = np.row_stack( [smin,smax] )
        self.orders = orders
        nodes = [np.linspace(smin[i], smax[i], orders[i]) for i in range(len(orders))]
        if len(orders) == 1:
            mesh = nodes
        else:
            mesh = np.meshgrid(*nodes)   # works only in 2d
            mesh = [m.flatten() for m in mesh]
    #        mesh.reverse()
        self.nodes = nodes
        self.grid = np.row_stack(mesh)

    def find_cell(self, x):
        self.domain = self
        inf = self.smin
        sup = self.smax
        indices = []
        for i in range(self.domain.d):
            xi =(x[i,:] - inf[i])/(sup[i]-inf[i])
            ni = numpy.floor( xi*self.orders[i] )
            ni = numpy.minimum(numpy.maximum(ni,0),self.orders[i]-1)
            indices.append(ni)
        indices = np.row_stack(indices)
        return indices

    def compute_density(self, x):
        cell_indices = self.find_cell(x)
        keep = numpy.isfinite( numpy.sum(cell_indices, axis=1) )
        cell_indices = cell_indices[keep,:]
        npoints = cell_indices.shape[1]
        counts = numpy.zeros(self.orders, dtype=numpy.int)

        #print cell_indices
        for j in range(cell_indices.shape[1]):
            ind = tuple( cell_indices[:,j] )
            counts[ind] += 1
        dens = counts/npoints
        return dens

class TriangulatedDomain:
    def __init__(self,points):
        from scipy.spatial import Delaunay
        self.d = points.shape[0]
        if self.d == 1:
            raise(Exception("Impossible to triangulate in 1 dimension."))
        self.grid = points
        self.delaunay = Delaunay(points.T)
        self.smin = numpy.min(points,axis=1)
        self.smax = numpy.max(points,axis=1)
        self.bounds = np.array( [self.smin,self.smax] )
        
        

def SplineInterpolation(smin,smax,orders):
    if len(orders) == 1:
        return SplineInterpolation1(orders,smin,smax)
    elif len(orders) == 2:
        return SplineInterpolation2(orders,smin,smax)

class SplineInterpolation1:

    grid = None
    __values__ = None

    def __init__(self, orders, smin, smax):
        order = orders[0]
        smin = smin[0]
        smax = smax[0]
        grid = np.row_stack([np.linspace(smin, smax, order)])
        self.grid = grid
        pass

    def __call__(self,points):
        return self.interpolate(points)[0]

    def set_values(self,val):
        from scipy.interpolate import InterpolatedUnivariateSpline
        self.__values__ = val
        fgrid = self.grid.flatten()
        self.__splines__ = [ InterpolatedUnivariateSpline(fgrid, val[i,:]) for i in range(val.shape[0]) ]

    def interpolate(self, points, with_derivatives=False, with_coeffs_derivs=False):
        n_v = self.__values__.shape[0]
        n_p = points.shape[1]
        fpoints = points.flatten()
        val = np.zeros((n_v,n_p))
        dval = np.zeros((n_v,1,n_p))

        if with_coeffs_derivs:
            import time
            time1 = time.time()
            eps = 1e-5
            original_values = self.__values__
            resp = np.zeros((n_v,n_v,n_p))
            for i in range(n_v):
                new_values = original_values.copy()
                new_values[i,:] += eps
                self.set_values(new_values)
                resp[:,i,:] = self.interpolate(points,with_derivatives=False,with_coeffs_derivs=False)
            time2 = time.time()
            print('Derivative computation : ' + str(time2-time1))


        for i in range(self.__values__.shape[0]):
            spline = self.__splines__[i]
            y = spline(fpoints)
            dy = spline(fpoints,1)
            val[i,:] = y
            dval[i,0,:] = dy

        if not with_derivatives:
            return val
        else:
            return [val,dval]

class SplineInterpolation2:

    grid = None
    __values__ = None

    def __init__(self, orders, smin, smax):
        self.d = 2
        nodes = [np.linspace(smin[i], smax[i], orders[i]) for i in range(len(orders))]
#        nodes.reverse()
        mesh = np.meshgrid(*nodes)
        mesh = [m.flatten() for m in mesh]
#        mesh.reverse()
        self.nodes = nodes
        self.grid = np.row_stack(mesh)
        pass
    
    def __call__(self,points):
        return self.interpolate(points)[0]

    def set_values(self,val):
        from scipy.interpolate import RectBivariateSpline
        self.__values__ = val
#        fgrid = self.grid.flatten()
        [grid_x, grid_y] = self.nodes
#        grid_x = self.grid[0,:]
#        grid_y = self.grid[1,:]
#        print grid_x
#        print grid_y
        n_x = len(grid_x)
        n_y = len(grid_y)
        # TODO: options to change 0.1
        margin = 0.5
        bbox = [min(grid_x)- margin, max(grid_x) + margin,min(grid_y)-margin, max(grid_y) + margin]

        self.__splines__ = [ RectBivariateSpline( grid_x, grid_y, val[i,:].reshape((n_y,n_x)).T, bbox=bbox ) for i in range(val.shape[0]) ]

    def interpolate(self, points, with_derivatives=False):
        n_v = self.__values__.shape[0]
        n_p = points.shape[1]
        n_d = self.d
#        fpoints = points.flatten()

        val = np.zeros((n_v,n_p))
        dval = np.zeros((n_v,n_d,n_p))

        for i in range(self.__values__.shape[0]):
            spline = self.__splines__[i]
            A = points[0,:] # .reshape((n_x,n_y))
            B = points[1,:] # .reshape((n_x,n_y))

            eps = 0.0001
            y = spline.ev(A, B)
            d_y_A = ( spline.ev(A + eps, B) - y ) / eps
            d_y_B = ( spline.ev(A, B + eps) - y ) / eps

            val[i,:] = y
            dval[i,0,:] = d_y_A

            dval[i,1,:] = d_y_B

        if not with_derivatives:
            return val
        else:
            return [val,dval]

class LinearTriangulation:
    def __init__(self,domain):
        self.domain = domain
        self.delaunay = domain.delaunay

    def __call__(self, zz):
        return self.interpolate(zz)[0]


    def set_values(self, val):
        self.__values__ = val

    def interpolate(self, points, with_derivatives=False):
        n_x = self.__values__.shape[0]
        n_p = points.shape[1]
        n_d = self.domain.d
        resp = np.zeros((n_x,n_p))
        dresp = np.zeros((n_x,n_d,n_p))
        for i in range(n_x):
            [val,dval] = self.interpolate_1v(i,points)
            resp[i,:] = val
            dresp[i,:,:] = dval
        if not with_derivatives:
            return resp
        else:
            return [resp,dresp]

    def interpolate_1v(self, i, points):

#        points = numpy.minimum(points, self.domain.smax) # only for rectangular domains
#        points = numpy.maximum(points, self.domain.smin) # only for rectangular domains
#        print points

        zz = points
        ndim = self.domain.d
        delaunay = self.delaunay
        nvalues_on_points = self.__values__[i,:]
        from dolo.numeric.serial_operations import serial_dot
        n_p = zz.shape[1]
        n_x = zz.shape[0]
        resp = numpy.zeros(n_p)
        dresp = numpy.zeros( (n_x, n_p) )
        inds_simplices = delaunay.find_simplex(zz.T)
        inside = (inds_simplices != -1)
        if True not in inside:
            return [resp,dresp]
        indices = inds_simplices[inside]
        transform = delaunay.transform[indices,:,:]
        transform = numpy.rollaxis(transform,0,3)
        Tinv = transform[:ndim,:ndim,:]
        r = transform[ndim,:,:]
        vertices = delaunay.vertices.T[:,indices]

        z = zz[:,inside]

        values_on_vertices = nvalues_on_points[vertices]

        last_V = values_on_vertices[-1,:]
        D = values_on_vertices[:-1,:] - last_V
        z_r = z-r
        c = serial_dot(Tinv, z_r)
        interp_vals = serial_dot(c, D) + last_V
        interp_dvals = serial_dot(D, Tinv)

        resp[inside] = interp_vals
        dresp[:,inside] = interp_dvals

        return [resp,dresp]




class MLinInterpolation:
    # piecewise linear interpolation

    grid = None
    __values__ = None

    def __init__(self, smin, smax, orders):
        nodes = [np.linspace(smin[i],smax[i],orders[i]) for i in range(len(orders))]
        grid = cartesian(nodes).T
        self.grid = grid
        self.__nodes__ = nodes
        pass

    def set_values(self,val):
        self.__values__ = val
        self.__coeffs__ = val ##

    def interpolate(self, points, with_derivatives=False):

        eps = 1e-6
        points_T = points.T
        
        from scipy.interpolate import LinearNDInterpolator

        interp = LinearNDInterpolator( self.grid.T, self.__values__.T )
        val0 = interp(points_T)

        if not with_derivatives:
            return val0.T
        else:
            # compute derivatives
            n_p = points_T.shape[0]
            n_d = points_T.shape[1]
            args_1 = [ points_T+np.tile(l,(n_p,1)) for l in np.eye(n_d)*eps]
            args_2 = [ points_T+np.tile(l,(n_p,1)) for l in -np.eye(n_d)*eps]


            dval1 = interp(args_1)
            dval2 = interp(args_2)


            dval = np.empty_like(dval1)
            for i in range(n_d):
                dd1 = (1- np.isnan(dval1[i,:,:])) * np.nan_to_num(dval1[i,:,:]) + np.isnan(dval1[i,:,:]) * val0
                dd2 = (1- np.isnan(dval2[i,:,:])) * np.nan_to_num(dval2[i,:,:]) + np.isnan(dval2[i,:,:]) * val0
                bound = np.isnan(dval1[i,:,:]) | np.isnan(dval2[i,:,:])
                neps = bound + (1-bound)*2
                #neps = 1/neps
                dval[i,:,:] = (dd1-dd2)/(eps)/neps
                #dval[i,:,:] = (dd1-dd2)/(2*eps)
            #now the ordering of dval is: (deriv, point, outdim)
            # reorder before sending back: (outdim,deriv,point)
            dval = np.rollaxis(dval,2,0)
            return [val0.T,dval]





if __name__ =='__main__':

    ## test splines
    from numpy import * 
    beta = 0.96
    bounds = array( [[-1], [1]] )
    orders = array( [10] )
    d = bounds.shape[1]
#
#    smin = bounds[0,:]
#    smax = bounds[1,:]
#    interp = SplineInterpolation(orders, smin, smax)
#    grid = interp.grid
#
#    vals = np.sin(grid)
#    interp.set_values(vals)
#
#    xvec = linspace(-1,1,100)
#    xvec = np.atleast_2d(xvec)
#    yvec = interp(xvec)
#
#
#    from matplotlib.pyplot import *
#    plot(interp.grid.flatten(), vals.flatten(),'o')
#    plot(xvec.flatten(),yvec.flatten())
    #show()


    from dolo.numeric.quantization import standard_quantization_weights
    from matplotlib import pyplot

    f = lambda x: 1 - x[0:1,:]**2 - x[1:2,:]**2

    ndim = 2
    N = 10
    [weights, points] = standard_quantization_weights( N, ndim )

    domain = TriangulatedDomain(points)
    interp = LinearTriangulation(domain)

    values = f(domain.grid)

    interp.set_values(values)


    orders = [100,100]
    smin = domain.smin
    smax = domain.smax
    extent = [smin[0], smax[0], smin[1], smax[1]]
    recdomain = RectangularDomain(smin,smax,orders)
    true_values = f( recdomain.grid )
    linapprox_values = interp(recdomain.grid)

    points = domain.grid

    pyplot.figure()
    pyplot.subplot(221)
#    pyplot.axes().set_aspect('equal')
    pyplot.imshow( true_values.reshape(orders), extent=extent,origin='lower', interpolation='nearest' )

    pyplot.plot(points[0,:],points[1,:],'o')
    pyplot.colorbar()
    pyplot.subplot(222)
    print linapprox_values.shape
    pyplot.imshow(linapprox_values.reshape(orders))
    pyplot.colorbar()
    pyplot.subplot(223)
    pyplot.imshow( (true_values- linapprox_values).reshape(orders))
    pyplot.colorbar()


    pyplot.show()
    exit()

    pyplot.figure()




    minbound = delaunay.min_bound
    maxbound = delaunay.max_bound
    extent = [minbound[0],maxbound[0],minbound[1],maxbound[1]]
    pyplot.bone()
    pyplot.figure()
    pyplot.imshow( values, extent=extent, origin='lower' )
    pyplot.colorbar()
    pyplot.figure()
    pyplot.imshow( interp_values, extent=extent,origin='lower', interpolation='nearest' )
    pyplot.axes().set_aspect('equal')
    for el in triangles: #triangulation.get_elements():
        plot_triangle(el)
    pyplot.colorbar()
    pyplot.figure()
    pyplot.imshow( abs(interp_values - values), extent=extent,origin='lower' )
    pylab.colorbar()

    triangles = []
    for v in delaunay.vertices:
        pp = [points[e,:] for e in v]
        triangles.append(pp)
    def plot_triangle(tr):
        ttr = tr + [tr[0]]
        ar = numpy.array(ttr)
        pyplot.plot(ar[:,0],ar[:,1], color='black')
    pyplot.figure()
    pyplot.axes().set_aspect('equal')
    pyplot.plot(points[:,0],points[:,1],'o')
    for el in triangles: #triangulation.get_elements():
        plot_triangle(el)
