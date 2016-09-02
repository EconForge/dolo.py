import numpy as np
import scipy.sparse as spa
from dolo.algos.dtcscc.time_iteration import time_iteration, time_iteration_direct
from dolo.numeric.misc import mlinspace
from dolo.numeric.discretization.discretization import rouwenhorst
from scipy.optimize import brentq, ridder, bisect
import warnings


warnings.warn("The distributions module is experimental. Please report any errors you may encounter during use to the EconForge/dolo issue tracker.")

# # Check whether inverse transition is in the model.
# ('transition_inv' in model.functions)

def stat_dist(model, dr, Nf, itmaxL=5000, tolL=1e-8, verbose=False):
    '''
    Compute a histogram of the stationary distribution for some fixed set of
    aggregate variables.

    Parameters
    ----------
    model : NumericModel
        "dtcscc" model to be solved
    dr : Decision rule
        Decision rule associated with solution to the model
    Nf : array
        Number of fine grid points in each dimension
    itmaxL : int
        Maximum number of iterations over the distribution evolution equation
    tolL : int
        Tolerance on the distance between subsequent distributions

    Returns
    -------
    L : array
        The density across states for the model. Note, the implied
        grid that L lays on follows the convention that for [N1, N2, N3, ...],
        earlier states vary slower than later states.
    QT : array
        The distribution transition matrix, i.e. L' = QT*L
    '''

    # HACK: get number of exogenous states from the number of shocks in
    # the model. We are assuming that each shock is associated with an
    # exogenous state. That is, no IID shocks enter the model on their own.
    Nexo = len(model.calibration['shocks'])
    Nend = len(model.calibration['states']) - Nexo

    # Total number of continuous states
    Ntot = np.prod(Nf)

    # Create fine grid for the histogram
    grid = fine_grid(model, Nf)
    parms = model.calibration['parameters']

    # Find the state tomorrow on the fine grid
    sprimef = dr_to_sprime(model, dr, Nf)

    # Compute exogenous state transition matrices
    mgrid, Qm = exog_grid_trans(model, Nf)

    # Compute endogenous state transition matrices
    # First state:
    sgrid = np.unique(grid[:,Nexo])
    Qs = single_state_transition_matrix(sgrid, sprimef[:,Nexo], Nf, Nf[Nexo]).toarray()

    # Subsequent state transitions created via repeated tensor products
    for i_s in range(1,Nend):
        sgrid = np.unique(grid[:,Nexo + i_s])
        Qtmp = single_state_transition_matrix(sgrid, sprimef[:,Nexo + i_s], Nf, Nf[Nexo + i_s]).toarray()
        N = Qs.shape[1]*Qtmp.shape[1]
        Qs = Qs[:, :, None]*Qtmp[:, None, :]
        Qs = Qs.reshape([N, -1])

    # Construct all-state transitions via endogenous-exogenous tensor product
    Q = Qm[:, :, None]*Qs[:, None, :]
    Q = Q.reshape([Ntot, -1])
    QT = spa.csr_matrix(Q).T
    # TODO: Need to keep the row kronecker product in sparse matrix format

    # Iterate over distribution transition equation until convergence
    L = iter_dist(QT, itmaxL=itmaxL, tolL=tolL, verbose=verbose)

    return L, QT

def iter_dist(QT, itmaxL, tolL, verbose=False):
    '''
    Given a distribution transition matrix, QT, start with an arbitrary
    distribution and iterate until convergence, i.e. until a stationary
    distribution is found.

    Parameters
    ----------
    QT : sparse matrix
        Transpose of the transition matrix describing transitions across
        distributions.
    itmaxL : int
        Maximum number of iterations over the distribution evolution equation
    tolL : int
        Tolerance on the distance between subsequent distributions

    Returns
    -------
    L : array
        The density across states for the model. Note, the implied
        grid that L lays on follows the convention that for [N1, N2, N3, ...],
        earlier states vary slower than later states.
    '''
    # Length of distribution
    N = QT.shape[0]

    # Start from uniform distribution
    L = np.ones(N)
    L = L/sum(L)

    # Iterate over the transition rule L' = QT@L until convergence
    for itL in range(itmaxL):
        Lnew = QT@L
        dL = np.linalg.norm(Lnew-L)/np.linalg.norm(L)
        if (dL < tolL):
            break

        L = np.copy(Lnew)

        if verbose is True:
            if np.mod(itL, 100) == 0:
                print('Iteration = %i, dist = %f \n' % (itL, dL))

    return L




def single_state_transition_matrix(grid, vals, Nf, Nstate):
    '''
    Compute the transition matrix for an individual state variable.
    Transitions are from states defined on a fine grid with the dimensions
    in Nf, to the unique values that lie on the individual state grid
    (with dimension Nstate).

    Parameters
    ----------
    grid : Array
        The approximated model's state space defined on a grid. Must be unique
        values, sorted in ascending order.
    vals : Array
        Actual values of state variable next period (computed using a
        transition or decision rule)
    Nf : array
        Number of fine grid points in each dimension
    Nstate : int
        Number of fine grid points for the state variable in question.

    Returns
    -------
    Qstate : sparse matrix
        An [NtotxNstate] Transition probability matrix for the state
        variable in quesiton.

    '''
    Ntot = np.prod(Nf)

    idxlist = np.arange(0,Ntot)
    # Find upper and lower bracketing indices for the state values on the grid
    idL, idU = lookup(grid, vals)

    # Assign probability weight to the bracketing points on the grid
    weighttoupper = ( (vals - grid[idL])/(grid[idU] - grid[idL]) ).flatten()
    weighttolower = ( (grid[idU] - vals)/(grid[idU] - grid[idL]) ).flatten()

    # Construct sparse transition matrices.
    # Note: we convert to CSR for better sparse matrix arithmetic performance.
    QstateL = spa.coo_matrix((weighttolower, (idxlist, idL.flatten())), shape=(Ntot, Nstate))
    QstateU = spa.coo_matrix((weighttoupper, (idxlist, idU.flatten())), shape=(Ntot, Nstate))
    Qstate =(QstateL + QstateU).tocsr()

    return Qstate


# TODO: Need to allow for multiple aggregate variables to be computed.
# E.g. a model with aggregate capital and labor.
def solve_eqm(model, Nf, varname, aggvarname, bounds,
              method='damping', maxiter=100, toleq=1e-4, verbose=False):
    '''
    Solve for the equilibrium value of the aggregate capital stock in the
    model. Do this via a damping algorithm over the capital stock. Iterate
    until aggregate capital yields an interest rate that induces a
    distribution of capital accumulation across agents that is consistent
    with that capital stock.

    Parameters
    ----------
    model : NumericModel
        "dtcscc" model to be solved
    Nf : array
        Number of fine grid points in each dimension
    varname : string
        The string name of the agent-level variable. E.g. assets, consumption,
        labor, prices.
    aggvarname : string
        The string name of the aggregate variable. e.g
    bounds : tuple/list/array
        Bounds for the value of the aggregate variable
    method : string
        The method to solve for equilibrium (see scipy.optimize documentation):
        - damping
        - bisection
        - brent
        - ridder
    maxiter : int
        Maximum number of iterations over equilibrium computaiton algorithm
    toleq : int
        Maximum change in the equilibrium residual.

    Returns
    -------
    Aval : float
        Equilibrium value of the aggregate variable.
    '''

    if method is 'damping':
        Ainit = (bounds[0]+bounds[1])/2
        fun = lambda A: aggregate_resid(model, Nf, A, varname, aggvarname)
        Aval = damping(fun, Ainit, xtol=toleq, rtol=toleq, maxiter=100, verbose=verbose)

    elif method is 'bisection':
        fun = lambda A: aggregate_resid(model, Nf, A, varname, aggvarname)
        Aval = bisect(fun, bounds[0], bounds[1], xtol=toleq, rtol=toleq, maxiter=100, full_output=False, disp=verbose)

    elif method is 'brent':
        fun = lambda A: aggregate_resid(model, Nf, A, varname, aggvarname)
        Aval = brentq(fun, bounds[0], bounds[1], xtol=toleq, rtol=toleq, maxiter=100, full_output=False, disp=verbose)

    elif method is 'ridder':
        fun = lambda A: aggregate_resid(model, Nf, A, varname, aggvarname)
        Aval = ridder(fun, bounds[0], bounds[1], xtol=toleq, rtol=toleq, maxiter=100, full_output=False, disp=verbose)

    else:
        raise Exception("Other methods not supported.")

    return Aval



def damping(fun, Ainit, xtol=1e-6, rtol=1e-6, maxiter=100, verbose=False):
    '''
    Use damping method on the residuals of the aggregate equilibrium equation to
    find the equilibrium aggregate variable.

    Parameters
    ----------
    fun : function
        Function that yields the aggregate equation residual
    A0 : float
        Initial guess for the aggregate variable
    xtol : float
        Tolerance over distance between successive guesses
    rtol : float
        Tolerance over residual from the aggregate equation residual
    maxiter : int
        Maximum number iterations over the damping algorithm

    Returns
    -------
    A : float
        Equilibrium value of the aggregate variable.
    '''

    A_new = Ainit
    damp = 0.999

    for iteq in range(maxiter):
        A = A_new

        resid = fun(A)

        if verbose is True:
            print('Iteration = \t%i: resid=\t%1.4f' % (iteq, resid) )

        # Update guess for aggregate variable, and damping parameter
        A_new = A - (1-damp)*resid
        damp = 0.995*damp

        # Check tolerances
        if (np.abs(resid) < rtol):
            break
        if (np.abs(A - A_new) < xtol):
            break

    return A



# TODO: allow more general forms of the aggregate equation.
# At the moment can only handle equations of the form,
# e.g., K = integrate[ k'(k,e) d mu(k,e) ]
def aggregate_resid(model, Nf, Aval, varname, aggvarname):
    '''
    Solve for an aggregate state variable given

    Parameters
    ----------
    model : NumericModel
        "dtcscc" model to be solved
    dr : Decision rule
        Decision rule associated with solution to the model
    Nf : array
        Number of fine grid points in each dimension
    Aval : float
        Value of the aggregate state varaible
    varname : string
        The string name of the agent-level variable. E.g. assets, consumption,
        labor, prices.
    aggvarname : string
        The string name of the aggregate variable. e.g

    Returns
    -------
    resid : float
        The value of the residual from the aggregate condition:
        (guess for aggregate - aggregated).
    '''

    # Set calibration to current value of aggregate variable, solve for
    # decision rule, and compute the stationary distribution
    dr, L = set_solve_stat(model, Nf, aggvarname, Aval, verbose=False)

    # Set up
    states = model.symbolic.symbols['states']
    controls = model.symbolic.symbols['controls']

    # Locate agent-level variable in the model,
    # then compute aggregate value using distribution (L)
    grid = fine_grid(model, Nf)
    if varname in states:
        idx = states.index(varname)
        aggsum = np.dot(grid[:,idx], L)
    elif varname in controls:
        idx = controls.index(varname)
        aggsum = np.dot(dr(grid)[:,idx], L)

    resid = Aval - aggsum

    return resid


def set_solve_stat(model, Nf, aggvarname, Aval, verbose=False):
    '''
    Set aggregate calibration, solve the individual agent's decision rule,
    compute the stationary distribution.
        Parameters
        ----------
        model : NumericModel
            "dtcscc" model to be solved
        Nf : array
            Number of fine grid points in each dimension
        aggvarname : string
            The string name of the aggregate variable. e.g
        Aval : float
            Value of the aggregate state varaible

        Returns
        -------
        dr : Decision rule
            Decision rule associated with solution to the model
    L : array
        The density across states for the model. Note, the implied
        grid that L lays on follows the convention that for [N1, N2, N3, ...],
        earlier states vary slower than later states.
    '''
    # Set model calibration at given aggregate variable value
    model.set_calibration(aggvarname, Aval)

    # Solve for decision rule given current guess for aggregate
    if ('direct_response' in model.symbolic.equations):
        dr = time_iteration_direct(model, with_complementarities=True, verbose=verbose)
    else:
        dr = time_iteration(model, with_complementarities=True, verbose=verbose)

    # Solve for stationary distribution given decision rule
    L, QT = stat_dist(model, dr, Nf, verbose=verbose)

    return dr, L



# TODO: Modify to allow for non-straightforward aggregate residuals. So far can
#        only handle the form: resid = K - Kaggregated
def supply_demand(model, varname, aggvarname, pricename, Nf, bounds,
                  numpoints=20, verbose=True):
    '''
    Solve the model at a range of aggregate capital values to generate
    supply and demand curves a given aggregate variable.

    Parameters
    ----------
    model : NumericModel
        "dtcscc" model to be solved
    varname : string
        The string name of the agent-level variable. E.g. assets, consumption,
        labor, prices.
    aggvarname : string
        The string name of the aggregate variable. e.g
    pricename : string
        The string name of the price of the aggregate variable,
        e.g. price = 'r' picks out the interest rate
    Nf : array
        Number of fine grid points in each dimension
    lower : float
        Lower bound on aggregate variable (demand)
    upper : float
        Upper bound on aggregate variable (demand)
    numpoints : int
        Number of points at which to evaluate the curves

    Returns
    -------
    Ad : array
        Set of aggreate demands
    As : array
        Set of aggregate supplies
    p : array
        Set of prices at each point on the demand-supply curves
    '''
    # NOTE: This method only works if the residual is of the form:
    # resid = K - Kaggregated

    lower = bounds[0]
    upper = bounds[1]
    grid = fine_grid(model, Nf)

    Ad = np.linspace(lower,upper,numpoints)
    As = np.zeros([numpoints,1])
    p = np.zeros([numpoints,1])

    for i in range(numpoints):

        # Get the aggregate equation residuals
        resid = aggregate_resid(model, Nf, Ad[i], varname, aggvarname)

        # Get the price
        p[i] = model.calibration_dict[pricename]

        # Compute aggregated variable using the residual
        As[i] = (-1)*(resid - Ad[i])      # e.g.   As = (-1)*((K - Kagg) - K)

        if verbose is True:
            if np.mod(i, 5) == 0:
                print('Iteration = \t%i' % i)

    return Ad, As, p


def dr_to_sprime(model, dr, Nf):
    '''
    Solve the decision rule on the fine grid, and compute the next
    period's state variable.

    Parameters
    ----------
    model : NumericModel
        "dtcscc" model to be solved
    dr : Decision rule
        Decision rule associated with solution to the model
    Nf : array
        Number of fine grid points in each dimension

    Returns
    -------
    kprimef : array
        Next period's state variable, given the decision rule mdr
    '''

    # HACK: trick to get number of exogenous states
    Nexo = len(model.calibration['shocks'])

    gridf = fine_grid(model, Nf)
    trans = model.functions['transition']      # trans(s, x, e, p, out)
    parms = model.calibration['parameters']
    grid = model.get_grid()
    a = grid.a
    b = grid.b

    drc = dr(gridf)

    # Compute state variable transitions
    sprimef = trans(gridf, drc, np.zeros([1,Nexo]), parms)

    # Keep state variables on their respective grids
    sprimef = np.maximum(sprimef, a[None,:])
    sprimef = np.minimum(sprimef, b[None,:])

    return sprimef


def lookup(grid, x):
    '''
    Finds indices of points in the grid that bracket the values in x.
    Grid must be sorted in ascending order. Find the first index, i, in
    grid such that grid[i] <= x. This is the index of the upper bound for x,
    unless x is equal to the lowest value on the grid, in which case set the
    upper bound index equal to 1. The lower bound index is simply one less
    than the upper bound index.
    '''
    N = grid.shape[0]-1   # N = last index in grid
    m = grid.min()
    M = grid.max()
    x = np.maximum(x, m)
    x = np.minimum(x, M)
    idU = np.searchsorted(grid, x)   # Index of the upper bound
    idU = np.maximum(idU, 1)      # Upper bound is always greater than 1
    idL = idU -1                  # lower bound index = upper bound index - 1

    return idL, idU



def fine_grid(model, Nf):
    '''
    Construct evenly spaced fine grids for state variables. For endogenous
    variables use a uniform grid with the upper and lower bounds as
    specified in the yaml file. For exogenous variables, use grids from
    the discretization of the AR(1) process via Rouwenhorst.

    Parameters
    ----------
    model : NumericModel
        "dtcscc" model to be solved
    Nf : array
        Number of points on a fine grid for each endogeneous state
        variable, for use in computing the stationary distribution.

    Returns
    -------
    grid : array
        Fine grid for state variables. Note, exogenous ordered first,
        then endogenous. Later variables are "fastest" moving, earlier
        variables are "slowest" moving.
    '''

    # HACK: trick to get number of exogenous states
    Nexo = len(model.calibration['shocks'])
    Nend = len(model.calibration['states']) - Nexo

    # ENDOGENOUS VARIABLES
    grid = model.get_grid()
    a = grid.a
    b = grid.b
    sgrid = mlinspace(a[Nexo:],b[Nexo:],Nf[Nexo:])

    mgrid, Qm = exog_grid_trans(model, Nf)

    # Put endogenous and exogenous grids together
    gridf = np.hstack([np.repeat(mgrid, sgrid.shape[0],axis=0), np.tile(sgrid, (mgrid.shape[0],1))])

    return gridf

# TODO: incorporate this into multidimensional_discretization in
# dolo.numeric.discretization. Allow option to switch between the two assumptions
# (independent shocks or identical persistence).
# TODO: include a check to make sure that the covariance matrix is diagonal. If
# it is not, then use the multidimensional_discretization function.
def exog_grid_trans(model, Nf):
    '''
    Construct the grid and transition matrix for exogenous variables. Both
    elements are drawn from the Rouwenhorst descritization of the exogenous
    AR(1) process. The grids and transition matrices are compounded if
    there are multiple exogenous variables, and are constructed such that
    late variables are "fastest" moving, earlier variables are "slowest" moving.

    NOTE: This makes the strong assumption thhat the AR(1) processes are
    independent. In particular, the i.i.d shocks that drive each AR(1) process
    are assumed to be indpendent of each other (i.e. across AR(1) equations).

    This differs from multidimensional_discretization in
    dolo.numeric.discretization, which allows for correlated shocks, but assumes
    that the persistence parameter is identical across AR(1) equations.
    Parameters
    ----------
    model : NumericModel
        "dtcscc" model to be solved
    Nf : array
        Number of points on a fine grid for each endogeneous state variable,
        for use in computing the stationary distribution.

    Returns
    -------
    mgrid : array
        Fine grid for exogenous state variables.
    Qm : array
        Transition matrix (dimension: N x Nm)
    '''

    # HACK: trick to get number of exogenous states
    Nexo = len(model.calibration['shocks'])
    Nend = len(model.calibration['states']) - Nexo
    Nendtot = np.prod(Nf[Nexo:])

    distr = model.distribution
    sigma = distr.sigma

    # Check that ccovariance matrix is diagonal
    diags = sigma - np.diagonal(sigma)
    if not np.all(diags==0):
        raise Exception("Covariance matrix is not diagonal. Cannot proceed assuming that AR(1)s are indpendent.")

    # Get AR(1) persistence parameters via the derivative of the
    # transition rule at the steady state
    trans = model.functions['transition']    #  tmp(s, x, e, p, out)
    sss, xss, ess, pss = model.calibration['states', 'controls', 'shocks', 'parameters']
    diff = trans(sss, xss, ess, pss, diff=True)
    diff_s = diff[0]  # derivative wrt state variables

    # EXOGENOUS VARIABLES
    # Get first grid
    rho = diff_s[0]
    sig = np.sqrt(sigma[0,0])
    mgrid, Qm = rouwenhorst(rho, sig, Nf[0])
    mgrid = mgrid[:,None]

    # Get subsequent grids
    for i_m in range(1,Nexo):
        rho = diff_s[i_m]
        sig = np.sqrt(sigma[i_m,i_m])
        tmpgrid, Qtmp = rouwenhorst(rho, sig, Nf[i_m])
        # Compound the grids
        tmpgrid = np.tile(tmpgrid, mgrid.shape[0])
        mgrid = np.repeat(mgrid, Nf[i_m],axis=0)
        mgrid = np.hstack([mgrid, tmpgrid[:,None]])
        # Compound the transition matrices
        Qm = np.kron(Qm, Qtmp)

    # Repeat to match full sized state space.
    Qm = np.kron(Qm, np.ones([Nendtot,1]))

    return mgrid, Qm
