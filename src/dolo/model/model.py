# -*- coding: utf-8 -*-
from symbolic import *
import copy
import inspect
from sympy import Matrix


class Model:
    #commands = None # keep some instructions to treat the model, not sure how to do it
    def __init__(self, fname, equations, lookup=False):
        self.fname = fname
        self.variables = []
        #self.exovariables = []
        self.shocks = []
        self.parameters = []
        self.equations = []
        self.controls = []
        self.init_values = dict()
        self.parameters_values = dict()
        self.covariances = Matrix()
        self.variables_order = dict()
        self.variables_ordering = []
        self.parameters_ordering = []
        self.shocks_ordering = []
        self.equations = equations
        self.tags = {}
        if lookup:
            self.__lookup__()
        self.model = self # seems strange ! (for compatibility reasons)
        return(None)
    
    def __lookup__(self):
        """This function uses the context where Model() object has been created
        in order to initialize some member variables.
        """
        #initializer
        frame = inspect.currentframe().f_back.f_back
        try:
            #self.equations = frame.f_globals["equations"]
            self.variables_ordering = frame.f_globals["variables_order"]
            self.parameters_ordering = frame.f_globals["parameters_order"]
            self.shocks_ordering = frame.f_globals["shocks_order"]
        except:
            raise Exception(
"Dynare model : variables_order, shocks_order, or parameters_order have not been defined"
            )
        finally:
            del frame
            
    def copy(self):
        c = Model(self.fname)
        c.variables = copy.copy(self.variables)
        #c.exovariables = copy.copy(self.exovariables)
        c.covariances = copy.copy(self.covariances)
        c.shocks = copy.copy(self.shocks)
        c.parameters = copy.copy(self.parameters)
        c.equations = copy.copy(self.equations)
        c.init_values = copy.copy(self.init_values)
        c.commands = copy.copy(self.commands)
        return(c)

    def tag(self,h):
        self.tags.update(h)
        return(self)
    
    def check_all(self,verbose=False):
        print_info = verbose
        print_eq_info = verbose
        for i in range(len(self.equations)):
            self.equations[i].infos['n'] = i+1
            self.set_info(self.equations[i])
        info = {}
        if print_eq_info:
            for eq in self.equations:
                print("Eq (" + str(eq.n) +") : " + str(eq.info) )
        def multiunion(list_of_sets):
            res = set([])
            for s in list_of_sets:
                res = res.union(s)
            return res

        tv = list( multiunion( [ eq.info['vars'] for eq in self.equations ] ) )
        ts = list( multiunion( [ eq.info['shocks'] for eq in self.equations ] ) )
        tp = list( multiunion( [ eq.info['params'] for eq in self.equations ] ) )

        self.variables = self.reorder(tv,self.variables_ordering)
        self.shocks = self.reorder(ts,self.shocks_ordering)
        self.parameters = self.reorder(tp,self.parameters_ordering)


        info = {
                "n_variables" : len(self.variables),
                "n_variables" : len(self.variables),
                "n_shocks" : len(self.shocks),
                "n_equations" : len(self.equations)
        }
        self.info = info
        if True:
            print("Model check : " + self.fname)
            for k in info:
                print("\t"+k+"\t\t"+str(info[k]))

    def check(self,model_type="dynare",verbose=False):
        '''
        Runs a series of assertion to verify model compliance to uhlig/dynare conventions.
        '''
        if model_type == "uhlig":
            self.check_type_uhlig(verbose)
        elif model_type == "dynare":
            self.check_type_dynare(verbose)

    def check_type_dynare(self,verbose=False):
        self.check_all(verbose=verbose) #compute all informations
        return None

    def check_type_uhlig(self,verbose=False):
        self.check_all(verbose=verbose) #compute all informations
        try: assert(self.info['n_equations'] == self.info['n_variables'])
        except:
            raise Exception("Number of equations must equal number of variables and exovariables")
        # First we look for exogenous equations
        exo_eqs = [eq for eq in self.equations if eq.info.get('all_shocks')]
        exo_vars = set()
        for eq in exo_eqs:
            exo_v = eq.lhs
            try: assert( isinstance(exo_v,Variable) and (exo_v.lag == 0) and (exo_v in self.variables) )
            except: raise Exception('Exogenous equations left hand-side must contain only one variable at date t+1')
            exo_vars.add(exo_v)
        for eq in exo_eqs:
            eq.info['exogenous'] = 'true'
            vs = [ a for a in eq.rhs.atoms() if isinstance(a,Variable) ]
            rhs_shocks = [ v for v in vs if v in self.shocks ]
            rhs_vars = [ v for v in vs if v in self.variables ]
            shocks_lags = set([ s.lag for s in rhs_shocks ])
            try: assert( shocks_lags == set([0]))
            except:
                raise Exception('In exogenous equations, shocks should appear on the right hand side with lag 0 and coefficient 1.')
            # TODO : add the condition that coefficient must be equal to 1
            vars_lags = set([ v.lag for v in rhs_vars ])
            try: assert( vars_lags.issubset( set([-1])) )
            except:
                raise Exception('In exogenous equations, variables should appear on the right hand side with lag -1.')
            rhs_vars_c = set([ v.P for v in rhs_vars ])
            try: assert( rhs_vars_c.issubset( exo_vars ) )
            except:
                raise Exception("One variable has been found in an exogenous equation that doesn't seem to be exogenous")
    
    def set_info(self,eq):
        '''
        Computes all informations concerning one equation (leads , lags, ...)
        '''
        info = {}
        vars = set([])
        shocks = set([])
        params = set([])
        all_vars = set([]) # will contain all variables
        all_shocks = set([])
        for a in eq.atoms():
            if isinstance(a,Variable): # unnecessary
                vars.add(a.P)
                all_vars.add(a)
            elif isinstance(a,Shock):
                shocks.add(a.P)
                all_shocks.add(a)
            elif isinstance(a,Parameter):
                params.add(a)
        lags = [v.lag for v in all_vars]
        if len(lags) == 0:
            # this equations doesn't contain any variable
            info['constant'] = True
            eq.info = info
            return None
        else:
            info['constant'] = False
            # These information don't depend on the system of equations
            info['max_lag'] = max(lags)
            info['min_lag'] = min(lags)
            info['expected'] = (max(lags) > 0)
            # These information depend on the model
            #info['exogenous'] =set(all_vars_c).issubset(set(self.exovariables).union(self.shocks)) # & max(lags)<=0
        info['vars'] = vars
        info['all_vars'] = all_vars
        info['shocks'] = shocks
        info['all_shocks'] = all_shocks
        info['params'] = params
        eq.info = info
        
    def incidence_matrix_static(self):
        n = len(self.equations)
        mat = Matrix().zeros((n,n))
        for i in range(n):
            eq = self.equations[i]
            left_vars = set([v.P for v in eq.lhs.atoms() if isinstance(v,Variable)])
            right_vars = set([v.P for v in eq.rhs.atoms() if isinstance(v,Variable)])
            all_vars = left_vars.union(right_vars)
            for v in all_vars:
                j = self.variables.index(v)
                if v in left_vars and v in right_vars:
                    mat[i,j] = 2
                elif v in left_vars:
                    mat[i,j] = -1
                elif v in right_vars:
                    mat[i,j] = 1
        return( mat )
            

    def order_parameters_values(self):
        from dolo.misc.calculus import solve_triangular_system
        itp = dict()
        itp.update(self.parameters_values)
        porder = solve_triangular_system(itp,return_order=True)
        return porder

    def order_init_values(self):
        from dolo.misc.calculus import solve_triangular_system
        #[itp,porder] = self.solve_parameters_values
        itd = dict()
        itd.update(self.init_values)
        vorder = solve_triangular_system(itd,unknown_type=Variable,return_order=True)
        # should we include shocks initialization or not ?
        return vorder

        
        #itd.update(model.init_values)

    def reorder(self, vars, variables_order):
        arg = list(vars)
        res = []
        for v in variables_order:
            if isinstance(v,IndexedSymbol):
                name = v.basename
                l = []
                for h in vars:
                    if h in arg and name == h.father:
                        l.append(h)
                        arg.remove(h)
                l.sort()
                res.extend(l)
            else:
                name = v.name
                for h in vars:
                    if h in arg and h == v:
                        res.append(h)
                        arg.remove(h)
        for h in arg:
            res.append(h)
        return res

    def dss_equations(self):
        # returns all equations at the steady state
        from dolo.misc.calculus import map_function_to_expression
        c_eqs = []
        def f(x):
            if x in self.shocks:
                return(0)
            elif x.__class__ == Variable:
                return(x.P)
            else:
                return(x)
        for eq in self.equations:
            n_eq = map_function_to_expression(f,eq)
            #n_ecurrent_equationsq.is_endogenous = eq.is_endogenous
            c_eqs.append(n_eq)
        return(c_eqs)
    
    def future_variables(self):
        '''
        returns [f_vars, f_eq, f_eq_n]
        f_vars : list of variables with lag > 1
        f_eq : list of equations containing future variables
        f_eq_n : list of indices of equations containing future variables
        '''
        # this could be simplified dramatically
        f_eq_n = [] # indices of future equations
        f_eq = [] # future equations (same order)
        f_vars = set([]) # future variables
        for i in range(len(self.equations)):
            eq = self.equations[i]
            all_atoms = eq.atoms()
            f_var = []
            for a in all_atoms:
                if (a.__class__ == Variable) and (a(-a.lag) in self.variables):
                    if a.lag > 0:
                        f_var.append(a)
            if len(f_var)>0:
                f_eq_n.append(i)
                f_eq.append(eq)
                f_vars = f_vars.union(f_var)
        f_vars = list(f_vars)
        return([f_vars,f_eq,f_eq_n])


    @property
    def dyn_var_order(self):
        # returns a list of dynamic variables ordered as in Dynare's dynamic function
        d = dict()
        for eq in self.equations:
            all_vars = eq.variables
            for v in all_vars:
                if not v.lag in d:
                    d[v.lag] = set()
                d[v.lag].add(v)
        maximum = max(d.keys())
        minimum = min(d.keys())
        ord = []
        for i in range(minimum,maximum+1):
            if i in d.keys():
                ord += [v(i) for v in self.variables if v(i) in d[i]]

        return ord

    def compute_uhlig_matrices(self):
        model = self.model
        exo_eqs = [eq for eq in model.equations if eq.info.get('exogenous') == 'true']
        non_exo_eqs = [eq for eq in model.equations if not eq in exo_eqs]
        exo_vars = [eq.lhs for eq in exo_eqs]
        non_exo_vars = [v for v in model.variables if not v in exo_vars]
        self.info['exo_vars'] = exo_vars
        self.info['non_exo_vars'] = non_exo_vars

        mat_exo_vars_f = Matrix([v(+1) for v in exo_vars]).T
        mat_exo_vars_c = Matrix([v for v in exo_vars]).T
        mat_exo_vars_p = Matrix([v(-1) for v in exo_vars]).T

        mat_non_exo_vars_f = Matrix( [v(+1) for v in non_exo_vars] ).T
        mat_non_exo_vars_c = Matrix( [v for v in non_exo_vars] ).T
        mat_non_exo_vars_p = Matrix( [v(-1) for v in non_exo_vars] ).T

        # Compute matrix for exogenous equations
        mat_exo_rhs = Matrix([eq.rhs for eq in exo_eqs]).T
        N = mat_exo_rhs.jacobian(mat_exo_vars_p).T

        # Compute matrices for non exogenous equations
        mat_non_exo_eqs = Matrix( [ eq.gap() for eq in non_exo_eqs ] ).T
        F = mat_non_exo_eqs.jacobian(mat_non_exo_vars_f).T
        G = mat_non_exo_eqs.jacobian(mat_non_exo_vars_c).T
        H = mat_non_exo_eqs.jacobian(mat_non_exo_vars_p).T
        L = mat_non_exo_eqs.jacobian(mat_exo_vars_f).T
        M = mat_non_exo_eqs.jacobian(mat_exo_vars_c).T

        def steady_state_ify(m):
            # replaces all variables in m by steady state value
            for v in self.variables + self.shocks: # slow and inefficient
                m = m.subs(v(+1),v.P)
                m = m.subs(v(-1),v.P)
            return m

        d = dict()
        d['F'] = steady_state_ify(F)
        d['G'] = steady_state_ify(G)
        d['H'] = steady_state_ify(H)
        d['L'] = steady_state_ify(L)
        d['M'] = steady_state_ify(M)
        d['N'] = steady_state_ify(N)
        return d


# for compatibility purposes
UhligModel = Model
DynareModel = Model