from dolo import *
#from misc import cal

set_variables("k c z")
set_shocks("x")
set_parameters("alpha beta gamma delta aa")

equations = [
    Equation(c + k , (1 - delta)*k(-1) + k(-1)**alpha*(1 + x)*aa),
    Equation(0 , -1/(1 + beta)*c(1)**(-gamma)*(1 - delta + k**(-1 + alpha)*(1 + z(1))*aa*alpha) + c**(-gamma)),
    Equation(z,x)
]

parameters_values = {
    alpha : 0.5,
    gamma : 0.5,
    delta : 0.02,
    beta : 0.05,
    aa : 0.5
}

covariances = Matrix([[0.01]])

init_values = {
    k:(1/aa/alpha*(beta + delta))**(1/(-1 + alpha)),
    c:k**alpha*aa - delta*k
}

############################


model = Model("pyramst",lookup=True)

#model.check_dynare()
model.check()


print model.order_parameters_values()
print model.order_init_values()

#model.check_uhlig()
#d = model.compute_formal_matrices()


#for k in d.keys():
    #print(k + ' : ')
    #print d[k]

###########################

from dolo.compiler.compiler_dynare import *

comp = DynareCompiler(model)

#print comp.export_to_modfile()

#print comp.main_file()

print comp.compute_dynamic_mfile(max_order=3)

