txt = '''
declarations:

    variables: [N_po, N_ce, C_po, C_ce, Z_ce, Z_po, Welf_po, Welf_ce, rho_po, rho_ce, mu_po, mu_ce, epsilon_po, epsilon_ce, Z]
    shocks: [e_Z]
    parameters: [theta, xi, sigma, beta, delta, alpha, rho_Z, CC_po, CC_ce, CC_tr, K, Nbar]

equations:
  - Welf_po=log(C_po)+beta*Welf_po(+1)
  - Welf_ce=log(C_ce)+beta*Welf_ce(+1)

  # Firm dynamics
  - N_po=(1-delta)*(N_po(-1)+Z-C_po/rho_po(-1))
  - N_ce=(1-delta)*(N_ce(-1)+Z-C_ce/rho_ce(-1))

  # Euler equation
  - rho_po(-1)*(C_po^(-1))=beta*(1-delta)*C_po(+1)^(-1)*(rho_po + C_po(+1)/N_po*epsilon_po)
  - rho_ce(-1)*(C_ce^(-1))=beta*(1-delta)*C_ce(+1)^(-1)*(rho_ce*mu_ce(-1)/mu_ce+C_ce(+1)/N_ce*mu_ce(-1)*(1-1/mu_ce) )

  - mu_po = 1+1/alpha/N_po
  - rho_po = exp(-0.5*(Nbar-N_po)/alpha/Nbar/N_po)
  - epsilon_po = (mu_po-1)/2


  - mu_ce = 1+1/alpha/N_ce
  - rho_ce = exp(-0.5*(Nbar-N_ce)/alpha/Nbar/N_ce)
  - epsilon_ce = (mu_ce-1)/2

  - log(Z)=rho_Z*log(Z(-1))+e_Z

calibration:

  parameters:
    theta: 6
    xi: 1.41
    sigma: 1
    beta: 0.99
    delta: 0.025
    rho_Z: 0.95
    Nbar: 1
    alpha: 1
    K: 1/beta/(1-delta) - 1
    CC_po: (1.0 - 1/beta/(1-delta) ) / (1-xi)
    CC_ce: (1.0 - 1/beta/(1-delta) ) / (1+theta/(1-theta))
#    alpha: 1/(xi-1)/N_po


  steady_state:
    Z: 1.0

    N_po: (1-delta)*Z/(delta+(1-delta)*CC_po)
    C_po:  CC_po * N_po^(xi)

    N_ce:  1 / ( delta/(1-delta)/2 + sqrt( (delta/(1-delta))^2 + 4*K*alpha ) / 2 )
    C_ce:  CC_ce * N_ce^(xi)

    mu_po: 1+1/alpha/N_po
    rho_po: exp(-0.5*(Nbar-N_po)/alpha/Nbar/N_po)
    epsilon_po: (mu_po-1)/2


    mu_ce: 1+1/alpha/N_ce
    rho_ce: exp(-0.5*(Nbar-N_ce)/alpha/Nbar/N_ce)
    epsilon_ce: (mu_ce-1)/2



    Welf_po: log(C_po)/(1-beta)
    Welf_ce : log(C_ce)/(1-beta)

  covariances: [[0.01]]
'''

from dolo import *

from dolo.misc.yamlfile import parse_yaml_text

model = parse_yaml_text(txt)

dr = solve_decision_rule(model)

