from __future__ import division

import yaml
import sympy



with file('../../../examples/two_countries.yaml') as f:
    txt = f.read()

data = yaml.load(txt)

#######################################

import numpy as np


from dolo.symbolic.symbolic import Variable, Parameter, Shock, Equation

declarations = data['declarations']

state_variables = [Variable(s,0) for s in declarations['state_variables']]
control_variables = [Variable(s,0) for s in declarations['controls']]
shadow_1_variables = [Variable(s,0) for s in declarations['shadows_1']]
shadow_2_variables = [Variable(s,0) for s in declarations['shadows_2']]
exogenous_variables = [Variable(s,0) for s in declarations['exogenous']]
shocks = [Shock(s,0) for s in declarations['shocks']]
parameters = [Parameter(s) for s in declarations['parameters']]

shadow_variables = shadow_1_variables + shadow_2_variables

dictionary = dict()
for s in control_variables + shadow_1_variables + shadow_2_variables + exogenous_variables + shocks + parameters:
    dictionary[s.name] = s

covariances = sympy.Matrix(len(shocks),len(shocks),lambda i,j: Parameter('sigma_'+str(i)+'_'+str(j)))

calibration = data['calibration']

## parse equations

t_equations = data['equations']


equations = dict()
for eq_type in t_equations.keys():
    l = []
    for t_eq in t_equations[eq_type]:
        try:
            [t_lhs,t_rhs] = t_eq.split('=')
            lhs = eval(t_lhs,dictionary)
            rhs = eval(t_rhs,dictionary)
            l.append( Equation(lhs,rhs) )
        except Exception as e:
            print 'Error while evaluating : ' + t_eq
            raise e
    equations[eq_type] = l

# parse parameters
calibration = data['calibration']
values = {}
for k in calibration.keys():
    lhs = eval(k,dictionary)
    rhs = eval(str(calibration[k]),dictionary)
    values[lhs] = rhs
#print parameters_values

# parse init values
initval = data['init_values']
for k in initval.keys():
    lhs = eval(k,dictionary)
    rhs = eval(str(initval[k]),dictionary)
    values[lhs] = rhs

for e in exogenous_variables:
    values[e] = 1


from dolo.misc.calculus import solve_triangular_system
[numerical_values,order] = solve_triangular_system(values)

################################################# end of parsing


Sigma = np.eye(4)*0.001
R = np.eye(4)*0.5

from compiling import compile_function

args = [shadow_variables] + [[s(-1) for s in state_variables]] + [exogenous_variables] + [control_variables]
args_flat = []
for l in args: args_flat += l

parms_val = [numerical_values[v] for v in parameters]

f  = compile_function(equations['definitions_1']+equations['definitions_2'], args_flat, parameters, 3)

args_h =  [[s(1) for s in shadow_variables]] + [[s(1) for s in control_variables]]
args_h_flat =  [s for s in shadow_variables] + [s for s in control_variables] + [s(1) for s in shadow_variables] + [s(1) for s in control_variables]

h  = compile_function(equations['euler_capital']+equations['euler_pricing']+equations['euler_arbitrage'], args_h_flat, parameters, 3)

import scipy.optimize

##############

def compute_steady_shadows(K):

    x0 =  [numerical_values[v] for i,v in enumerate( shadow_variables ) ]


    exo_ss = [numerical_values[v] for i,v in enumerate( exogenous_variables ) ]
    control_ss = [K[i] for i,v in enumerate( control_variables ) ]
    states_ss = control_ss[:-3]

    ff = lambda x: f(np.concatenate([x,states_ss,exo_ss,control_ss]),parms_val)[0]
    resp = scipy.optimize.fsolve(ff,x0)
    return resp

def residuals(K,Z,nout=1):


    print 'Recomputing steady-state'
    S = compute_steady_shadows(K)

    numv = numerical_values.copy()
    for i,v in enumerate( control_variables ):
        numv[v] = K[i]
    for i,v in enumerate( shadow_variables):
        numv[v] = S[i]



    det_steady_state = np.array( [numv[v.P] for v in args_flat] )



    X = Z[:, :len( state_variables ) ]
    XX = X[:-3,:]   # TODO
    Y = Z[:, len(state_variables):]

    [f0,f1,f2,f3] = f(det_steady_state,parms_val)

    ind_y = [i for i,v in enumerate(args_flat) if v in shadow_variables]
    ind_u = [i for i,v in enumerate(args_flat) if v not in shadow_variables]

    from compiling import solve_phi, from_phi_to_psi


    [phi_1,phi_2,phi_3]=solve_phi([f0,f1,f2,f3],ind_y,ind_u)



    [psi_1,psi_2,psi_3] = from_phi_to_psi( [phi_1,phi_2,phi_3], Z)
    ind_psi_x = slice(0,len(state_variables))
    ind_psi_e = slice(len(state_variables), len(  state_variables ) + len(exogenous_variables) )

    #

    ind_1 = [i for i,v in enumerate(args_h_flat)  if v in shadow_variables]
    ind_2 = [i for i,v in enumerate(args_h_flat) if v in control_variables]
    ind_3 = [i for i,v in enumerate(args_h_flat) if v(-1) in shadow_variables]
    ind_4 = [i for i,v in enumerate(args_h_flat) if v(-1) in control_variables]
    ind_f = ind_3 + ind_4

    det_steady_state_h = [numv[v.P] for v in args_h_flat]
    
    #D_ss = det_steady_state_h[ind_3 + ind_4]
    [h0,h1,h2,h3] = h(det_steady_state_h,parms_val)


    from dolo.numeric.tensor import mdot,sdot

    ### compute static residual


    A = h0
    B = np.dot(h1[:,ind_3],  np.tensordot(  psi_2[:,ind_psi_e,:][:,:,ind_psi_e],  Sigma, axes = ((1,2),(0,1))   )   )

    V = np.row_stack([ Y, psi_1[:,ind_psi_e] ])

    C = 0.5*mdot( h2[:,ind_f,:][:,:,ind_f], [ V, V ]    )
    C = np.tensordot(C,Sigma)

    res = A + B + C

    if nout == 1:
        return [res]
    ### compute derivative of risky residual

    F_x = np.row_stack([
        psi_1[:,ind_psi_x],
        X,
        np.dot( psi_1[:,ind_psi_x], XX),
        np.dot( X, XX)
    ])

    F_E = np.row_stack([
        psi_1[:,ind_psi_e],
        Y,
        np.dot( psi_1[:,ind_psi_e], R ),
        np.dot( Y, R )
    ])

    A_x = np.dot( h1, F_x )
    A_E = np.dot( h1, F_E )


    B_ = sdot(h2[:,:,ind_3],  np.tensordot(  psi_2[:,ind_psi_e,:][:,:,ind_psi_e],  Sigma, axes = ((1,2),(0,1))   )   )
    B_x = sdot(B_,F_x)
    B_E = sdot(B_,F_E)

    C_ = 0.5*mdot( h3[:,:,ind_f,:][:,:,:,ind_f], [ V, V ]    )
    C_ = np.tensordot(C_,Sigma)
    C_x = sdot(C_,F_x)
    C_E = sdot(C_,F_E)

    cor_B_x = np.tensordot( psi_3[:,ind_psi_x,:,:][:,:,ind_psi_e,:][:,:,:,ind_psi_e], Sigma, axes = ((2,3),(0,1)))
    cor_B_x = sdot(h1[:,ind_3], cor_B_x)
    cor_B_x = sdot(cor_B_x, XX)

    cor_B_E = np.tensordot( psi_3[:,ind_psi_e,:,:][:,:,ind_psi_e,:][:,:,:,ind_psi_e], Sigma, axes = ((2,3),(0,1)))
    cor_B_E = sdot(h1[:,ind_3], cor_B_E)
    cor_B_E = sdot(cor_B_E, R)

    cor_C_x = sdot( h2[:,ind_3,:][:,:,ind_f], V)
    cor_C_x = np.tensordot( cor_C_x, psi_2[:, ind_psi_x,:][:,:,ind_psi_e], axes = ( 1,0)  )
    cor_C_x = np.tensordot( cor_C_x, Sigma, axes = ((1,3),(0,1)) )
    cor_C_x = np.dot(cor_C_x, XX)

    cor_C_E = sdot( h2[:,ind_3,:][:,:,ind_f], V)
    cor_C_E = np.tensordot( cor_C_E, psi_2[:, ind_psi_e,:][:,:,ind_psi_e], axes = ( 1,0)  )
    cor_C_E = np.tensordot( cor_C_E, Sigma, axes = ((1,3),(0,1)) )
    cor_C_E = np.dot(cor_C_E, R)


    res_x = A_x + B_x + C_x
    res_x = cor_B_x + cor_C_x

    res_E = A_E + B_E + C_E
    res_E = cor_B_E + cor_C_E

    dres = np.column_stack([res_x,res_E])

    return [res,dres]

#############################
#############################
#############################



K0 = [numerical_values[v] for v in control_variables]
Z0 = np.zeros( ( len(control_variables) , len( state_variables ) + len(exogenous_variables) )   )


dd =  residuals( np.array([ 1,1,0,0,0,5,5,5]), Z0)[0]
print residuals( np.array([ 1,1,0,0,0,5,5,5]), Z0)[0] - dd
print residuals( np.array([ 1.1,1,0,0,0,5,5,5]), Z0)[0] - dd
print residuals( np.array([ 1,1.1,0,0,0,5,5,5]), Z0)[0] -dd
print residuals( np.array([ 1,1,0,0,0,5,5,5]), Z0)[0] -dd
print residuals( np.array([ 1,1,0,0,0,5,5,5]), Z0)[0] - dd

fobj = lambda K: residuals(K,Z0)[0]

#import scipy.optimize
print scipy.optimize.fsolve(fobj,K0)