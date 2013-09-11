class GModel_fg_from_fga:

    model = None
    calibration = None
    functions = None
    symbols = None
    infos = dict()


    def __init__(self, model_fga):

        self.model = model_fga.model
        self.parent = model_fga

        self.model_type = 'fg'
    
        import copy

        self.symbols = copy.copy(self.parent.symbols)
#        self.symbols['controls'] = self.parent.symbols['controls'] + self.parent.symbols['auxiliary']
    
        self.__create_functions__()
        self.set_calibration({})

    def __create_functions__(self):
        ff = self.parent.functions['arbitrage']
        gg = self.parent.functions['transition']
        aa = self.parent.functions['auxiliary']
        from dolo.numeric.serial_operations import serial_multiplication as serial_mult
        def f(s,x,S,X,p,derivs=False):
            if derivs:
                [y,y_s,y_x] = aa(s,x,p,derivs=True)
                [Y,Y_S,Y_X] = aa(S,X,p,derivs=True)
                [r,r_s,r_x,r_y,r_S,r_X,r_Y] = ff(s,x,y,S,X,Y,p,derivs=True)
                r_s = r_s + serial_mult(r_y,y_s)
                r_x = r_x + serial_mult(r_y,y_x)
                r_S = r_S + serial_mult(r_Y,Y_S)
                r_X = r_X + serial_mult(r_Y,Y_X)
                return [r, r_s, r_x, r_S, r_X]
            y = aa(s,x,p)
            Y = aa(S,X,p)
            r = ff(s,x,y,S,X,Y,p)
            return r
    
        def g(s,x,e,p,derivs=False):
            if derivs:
                [y,y_s,y_x] = aa(s,x,p,derivs=True)
                [S,S_s,S_x,S_y,S_e] = gg(s,x,y,e,p,derivs=True)
                S_s = S_s + serial_mult(S_y,y_s)
                S_x = S_x + serial_mult(S_y,y_x)
                return [S,S_s,S_x,S_e]
            y = aa(s,x,p)
            S = gg(s,x,y,e,p)
            return S

        functions = dict(
                arbitrage = f,
                transition = g,
                auxiliary = aa
                )
        self.functions = functions

    @property
    def variables(self):
        vars = []
        for vg in self.symbols:
            if vg not in ('parameters','shocks'):
                vars.extend( self.symbols[vg] )
        return vars

    def set_calibration(self,*args,**kwargs):
        self.parent.set_calibration(*args,**kwargs)
        import copy
        from numpy import concatenate
        calibration = copy.copy(self.parent.calibration)
#        calibration['controls'] = concatenate( [calibration['controls'], calibration['auxiliary']] )
        self.calibration = calibration

    def get_calibration(self, name):
        return self.parent.get_calibration(name)

    @property
    def x_bounds(self):
        return self.parent.x_bounds

if __name__ == "__main__":
    from dolo import *
    model = yaml_import("examples/global_models/rbc.yaml")

    model_fg = GModel_fg_from_fga(model)

    print( model_fg.get_calibration(['alpha','rho']) )

    model_fg.set_calibration(alpha=0.1)
    print( model_fg.get_calibration(['alpha','rho']) )

    print(model_fg.x_bounds)

    from dolo.numeric.global_solve import global_solve

    dr = global_solve(model_fg, verbose=True)
    print(dr)
