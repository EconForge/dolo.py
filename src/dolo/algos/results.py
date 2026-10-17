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
    log: object = None  # TimeIterationLog
    trace: object = None  # {Nothing,IterationTrace}
    error_res: float = None

    @property
    def converged(self) -> bool:
        return bool(self.x_converged)

    @property
    def error_step(self) -> float:
        return self.err

    @property
    def error_eta(self) -> float:
        return self.err

    @property
    def error_eps(self) -> float:
        return self.error_res if self.error_res is not None else self.err


@dataclass
class EGMResult(AlgoResult):
    dr: object
    iterations: int
    dprocess: object
    a_converged: bool
    a_tol: float
    err: float
    # log: object  # TimeIterationLog
    # trace: object  # {Nothing,IterationTrace}


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
    radius: float = None  # :Float64
    trace_data: object = None
    L: object = None
    log: object = None
    trace: object = None

    @property
    def converged(self) -> bool:
        return bool(self.x_converged)

    @property
    def iterations(self) -> int:
        return self.N

    @property
    def error_res(self) -> float:
        return self.f_x

    @property
    def error_eps(self) -> float:
        return self.f_x

    @property
    def error_step(self) -> float:
        return self.d_x

    @property
    def error_eta(self) -> float:
        return self.d_x


NewtonResult = ImprovedTimeIterationResult


@dataclass
class PerturbationResult(AlgoResult):
    dr: object  #:BiTaylorExpansion
    generalized_eigenvalues: object  # :Vector
    stable: bool  # biggest e.v. lam of solution is < 1
    determined: bool  # next eigenvalue is > lam + epsilon (MOD solution well defined)
    unique: bool  # next eigenvalue is > 1
