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
    log: object # TimeIterationLog
    trace: object #{Nothing,IterationTrace}


@dataclass
class ValueIterationResult(AlgoResult):
    dr: object           #:AbstractDecisionRule
    drv: object          #:AbstractDecisionRule
    iterations: int
    dprocess: object     #:AbstractDiscretizedProcess
    x_converged: object  #:Bool
    x_tol: float
    x_err: float
    v_converged: bool
    v_tol: float
    v_err: float
    log: object          #:ValueIterationLog
    trace: object        #:Union{Nothing,IterationTrace}
