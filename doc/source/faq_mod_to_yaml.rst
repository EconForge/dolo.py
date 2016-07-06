How do I translate my Dynare-readable model file into a dolo-readable model file?
.................................................................................

Many macroeconomic models are readily solved in the Dynare package. These models
are typically written in ''mod'' files using the Dynare syntax. To solve these
models in Dolo, these ''mod'' files will have to be tanslated into ''yaml'' files
with the Dolo syntax. This section gives an example of how to translate the
standard RBC file.

Consider the RBC model as it may be written in a Dynare ''mod'' file:

.. code:: yaml

  % rbc.mod
  %----------------------------------------------------------------
  % 1. Defining variables
  %----------------------------------------------------------------

  var y c k i n w rk z;
  varexo e_z;

  parameters beta sigma eta chi delta alpha rho zbar sig_z;

  %----------------------------------------------------------------
  % 2. Calibration
  %----------------------------------------------------------------

  alpha   = 0.33;
  beta    = 0.99;
  delta   = 0.025;
  rho     = 0.80;
  sigma   = 1;
  sig_z   = 0.016;
  eta     = 1;
  chi     = 0.5;
  zbar    = 1;

  %----------------------------------------------------------------
  % 3. Model
  %----------------------------------------------------------------

  model;
    y = exp(z)*k(-1)^alpha*n^(1-alpha);
    c = y - i;
    rk = alpha*y/k(-1);
    w = (1-alpha)*y/n;
    c^(-sigma) = beta*( c(+1)^(-sigma)*( 1-delta+rk(+1) ) );
    w = chi*n^eta*c^sigma;
    z = (1-rho)*zbar + rho*z(-1) + e_z;
    k = (1-delta)*k(-1) + i;
  end;

  %----------------------------------------------------------------
  % 4. Computation
  %----------------------------------------------------------------

  initval;
    n = 0.33;
    z = zbar;
    e_z = 0;
    rk = 1/beta-1+delta;
    k = n/(rk/alpha)^(1/(1-alpha));
    w = (1-alpha)*exp(z)*(k/n)^(alpha);
    i = delta*k;
    y = exp(z)*k^alpha*n^(1-alpha);
    c = y - i;
  end;

  shocks;
  var e_z = sig_z^2;
  end;


The same model can be rewritten in Dolo syntax inside a yaml file:

.. code:: yaml

  # rbc.yaml
  name: Real Business Cycle

  model_type: dtcscc

  symbols:

     states:  [z, k]
     controls: [i, n]
     shocks: [e_z]
     parameters: [beta, sigma, eta, chi, delta, alpha, rho, zbar, sig_z]

  definitions:
      y: z*k^alpha*n^(1-alpha)
      c: y - i
      rk: alpha*y/k
      w: (1-alpha)*y/n

  equations:

      arbitrage:
          - 1 - beta*(c/c(1))^(sigma)*(1-delta+rk(1))  | 0 <= i <= inf
          - chi*n^eta*c^sigma - w                      | 0 <= n <= inf

      transition:
          - z = (1-rho)*zbar + rho*z(-1) + e_z
          - k = (1-delta)*k(-1) + i(-1)

  calibration:

      # parameters
      beta : 0.99
      delta : 0.025
      alpha : 0.33
      rho : 0.8
      sigma: 1
      eta: 1
      sig_z: 0.016
      zbar: 1
      chi : 0.5

      # endogenous variables
      n: 0.33
      k: n/(rk/alpha)^(1/(1-alpha))
      w: (1-alpha)*z*(k/n)^(alpha)
      i: delta*k
      y: z*k^alpha*n^(1-alpha)
      c: y - i
      z: zbar
      rk: 1/beta-1+delta

  options:

      distribution: !Normal
          sigma: [ [ sig_z**2] ]

Several differences between the model file types are worth pointing out.

  - Dolo model files explicitly specify which variables are states and which variables are controls. Dynare model files only make the distinction between endogenous and shock variables. In contrast to Dolo, Dynare model files often distinguish state variables by the timing convention adopted in the file. For example, in the RBC case, the capital variable is lagged: `k(-1)`.

  - Dolo model files explicitly specify arbitrage equations (i.e. the agent's optimality conditions) and transition equations for the state variables. Arbitrage equations may be accompanied by their associated complementarity conditions. These are not features of Dynare model files.

  - Dynare model files are sometimes written in their non-linear form (e.g. as is done here), and Dynare solves for the `order n` approximation. In these cases, if the shock process, `z`, is expressed in its linear form, then within model equations it needs to be expressed as an exponentiated varaible, `exp(z)`. This is not necessary in Dolo model files.

  - Dolo model files require that the distribution of the shock process be explicitly specified, whereas Dynare implicitly assumes that the distribution is normal.

For more details about the structure, components, and other features of dolo model files, see `the Dolo language`_.
.. modeling_language:
