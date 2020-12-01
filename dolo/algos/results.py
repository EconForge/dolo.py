class AlgoResult:
    pass


from dataclasses import dataclass


@dataclass
class TimeIterationResult(AlgoResult):
    dr: object
    iterations: int
    complementarities: bool
    dprocess: object
    x_converged: bool
    x_tol: float
    err: float
    log: object  # TimeIterationLog
    trace: object  # {Nothing,IterationTrace}


@dataclass
class ValueIterationResult(AlgoResult):
    dr: object  #:AbstractDecisionRule
    drv: object  #:AbstractDecisionRule
    iterations: int
    dprocess: object  #:AbstractDiscretizedProcess
    x_converged: object  #:Bool
    x_tol: float
    x_err: float
    v_converged: bool
    v_tol: float
    v_err: float
    log: object  #:ValueIterationLog
    trace: object  #:Union{Nothing,IterationTrace}


@dataclass
class ImprovedTimeIterationResult(AlgoResult):
    dr: object  #:AbstractDecisionRule
    N: int
    f_x: float  #:Float64
    d_x: float  #:Float64
    x_converged: bool  #:Bool
    complementarities: bool  #:Bool
    # Time_search::
    radius: float  # :Float64
    trace_data: object
    L: object


@dataclass
class PerturbationResult(AlgoResult):
    dr: object  #:BiTaylorExpansion
    generalized_eigenvalues: object  # :Vector
    stable: bool  # biggest e.v. lam of solution is < 1
    determined: bool  # next eigenvalue is > lam + epsilon (MOD solution well defined)
    unique: bool  # next eigenvalue is > 1
