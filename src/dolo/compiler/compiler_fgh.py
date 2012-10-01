
from dolo.symbolic.model import Model

class CModel_fgah:

    model_type = 'fgah'

    def __init__(self, model):

        self.model = model

        var_g = model['variables_groups']

        controls = var_g['controls']
        states = var_g['states']
        auxiliary = var_g['auxiliary']
        expectations = var_g['expectations']

        parameters = model.parameters
        shocks = model.shocks

        f_eqs =  model['equations_groups']['arbitrage']
        a_eqs =  model['equations_groups']['auxiliary']
        g_eqs =  model['equations_groups']['transition']
        h_eqs =  model['equations_groups']['expectation']

        f_eqs =  [eq.gap for eq in f_eqs]
        a_eqs =  [eq.rhs for eq in a_eqs]
        g_eqs = [eq.rhs for eq in g_eqs]
        h_eqs = [eq.rhs for eq in h_eqs]

        controls_f = [c(1) for c in controls]
        states_f = [c(1) for c in states]
        controls_p = [c(-1) for c in controls]
        states_p = [c(-1) for c in states]
        shocks_f = [c(1) for c in shocks]
        auxiliary_p = [c(-1) for c in auxiliary]
        auxiliary_f = [c(1) for c in auxiliary]


        args_g =  [states_p, controls_p, auxiliary_p, shocks]
        args_f =  [states, controls, auxiliary, expectations]
        args_a =  [states, controls]
        args_h =  [states_f, controls_f, auxiliary_f]

        from compiling import compile_multiargument_function

        self.__g__ = compile_multiargument_function(g_eqs, args_g, ['s','x','a','e'], parameters, 'g' )
        self.__f__ = compile_multiargument_function(f_eqs, args_f, ['s','x','a','z'], parameters, 'f' )
        self.__a__ = compile_multiargument_function(a_eqs, args_a, ['s','x'], parameters, 'a' )
        self.__h__ = compile_multiargument_function(h_eqs, args_h, ['s','x','a'], parameters, 'h' )

    def f(self,*args):
        return self.__f__(*args)

    def g(self,*args):
        return self.__g__(*args)

    def h(self,*args):
        return self.__h__(*args)

    def a(self,*args):
        return self.__a__(*args)

    def as_type(self,model_type):
        if model_type == 'fgah':
            return self
        elif model_type == 'fga':
            raise Exception("Conversion from type fgah to fga not implemented yet.")
        elif model_type == 'fg':
            return Model_fg_from_fgah(self)
        else:
            raise Exception('Model of type {0} cannot be cast to model of type {1}'.format(self.model_type, model_type))
        return

class CModel_fgh_from_fgah:

    model_type = 'fgh'

    def __init__(self, model_fgah):

        self.parent = model_fgah
        self.model = self.parent.model

    def g(self,s,x,e,p):
        a = self.parent.__a__(s,x,p)
        S = self.parent.__g__(s,x,a,e,p)
        return S

    def f(self,s,x,z,p):
        a = self.parent.__a__(s,x,p)
        F = self.parent.__f__(s,x,a,z,p)
        return F

    def h(self,s,x,p):
        a = self.parent.__a__(s,x,p)
        return self.parent.__h__(s,x,a,p)

    @property
    def sigma(self):
        return self.model.read_covariances()

    @property
    def parameters(self):
        return self.model.read_calibration()[2]


class Model_fg_from_fgah:

    def __init__(self, model_fgah):

        self.parent = model_fgah
        self.model = self.parent.model

    def g(self,s,x,e,p):
        a = self.parent.__a__(s,x,p)
        S = self.parent.__g__(s,x,a,e,p)
        return S

    def f(self,s,x,S,X,e,p):

        a = self.parent.__a__(s,x,p)
        A = self.parent.__a__(S,X,p)
        h = self.parent.__h__(S,X,A,p)
        F = self.parent.__f__(s,x,a,h,p)

        return F

    @property
    def sigma(self):
        return self.model.read_covariances()

    @property
    def parameters(self):
        return self.model.read_calibration()[2]


if __name__  == '__main__':

    from dolo.misc.yamlfile import yaml_import
    from dolo.numeric.perturbations_to_states import approximate_controls
    from dolo.numeric.global_solve import global_solve, global_solve_new


    model = yaml_import( '../../../examples/global_models/rbc_fgah.yaml')
    model_bis = yaml_import( '../../../examples/global_models/rbc.yaml')

    dr_pert = approximate_controls(model_bis)

    cm = CModel_fgah(model)


    cm_fg = cm.as_type('fg')


    import time

    t = time.time()
    for n in range(10):
        global_solve(model_bis, initial_dr=dr_pert, polish=False, interp_type='mlinear')
    s = time.time()
    print('Elapsed : {}'.format(s-t) )


    t = time.time()
    for n in range(10):
        global_solve_new(cm, initial_dr=dr_pert, polish=False, interp_type='mlinear')
    s = time.time()
    print('Elapsed : {}'.format(s-t) )