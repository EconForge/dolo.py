#from sympy import Symbol
from dolo.symbolic.symbolic import TSymbol

class DerivativesTree:

    symbol_type = TSymbol

    def __init__(self,expr,n=0,p=-1,vars=[],ref_var_list=None):
        self.expr = expr
        self.n = n
        self.p = p
        self.vars = vars
        self.derivatives = dict()
        # if index is not given it is constructed
        if ref_var_list == None:
            # TODO: remove this
            self.ref_var_list = [s for s in expr.atoms() if isinstance(s,self.symbol_type)]
        else:
            self.ref_var_list = ref_var_list

    def compute_children(self):
        self.derivatives = dict()
        #deriv_vars = [s for s in self.expr.atoms() if isinstance(s,self.symbol_type)]
        deriv_vars = [s for s in self.expr.atoms() if s in self.ref_var_list]
        for v in deriv_vars:
            i = self.ref_var_list.index(v)
            if i >= self.p:
                ndt = DerivativesTree(self.expr.diff(v), self.n + 1, i, self.vars + [v], self.ref_var_list)
                self.derivatives[i] = ndt
        return self.derivatives

    def compute_nth_order_children(self,n):
        if n > 0:
            self.compute_children()
            for c in self.derivatives.values():
                c.compute_nth_order_children(n-1)

    def compute_index_set(self,order_list):
        counts = [self.vars.count(v) for v in order_list]
        permutations = permutations_with_repetitions(counts)
        return permutations

    def compute_index_set_matlab(self,order_list):
        n = len(order_list)
        permutations = self.compute_index_set(order_list)
        np = len(permutations[0])
        perms_matlab = []
        for perm in permutations:
            index = 1 + sum_list( [ (perm[i]) * ((n))**(i)  for i in range(np) ] )
            perms_matlab.append(index)
        return perms_matlab

    def depth(self):
        if len(self.derivatives) == 0:
            return 0
        else:
            dep = 1 + max( [  ndt.depth() for ndt in self.derivatives.values() ] )
            return dep


    def list_nth_order_children(self,n):
        if n == 0:
            return [self]
        else:
            l = list()
            for c in self.derivatives.values():
                l.extend(c.list_nth_order_children(n-1))
            return l

def sum_list(l):
    s = 0
    for e in l:
        s += e
    return s

def permutations_with_repetitions(items):
    if sum_list(items) == 0:
        return [[]]
    else:
        resp = []
        for i in range(len(items)):
            if items[i]>0:
                newitems = list(tuple(items))
                newitems[i] -= 1
                resp.extend(  [ [i] + e for e in permutations_with_repetitions(newitems)] )
        return resp