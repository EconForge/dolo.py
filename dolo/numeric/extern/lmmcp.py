# THis optimizer is a python port of the original LMMCP software
# written by Christian Kanzow and Stefania Petra
# Original source code and documentation can be found at:
#  http://www.mathematik.uni-wuerzburg.de/~kanzow/

# ported from lmmcp.m
import time
import numpy as np

# parameter settings
eps1 = 1e-8    # default: 1e-6
eps2 = 1e-12   # default: 1e-10
null = 1e-8    # default: 1e-8
Big  = 1e10    # default: 1e+10

# maximal number of iterations
kmax=500   # default: 500

# choice of lambda
lambda1=0.1 # default: 0.1
lambda2=1-lambda1

# steplength parameters
beta = 0.55   # default: 0.55
sigma = 1e-4  # default: 1e-4
tmin = 1e-12  # default: 1e-12

rho=1e-8      # default: 1e-8
p=2.1         # default: 2.1

# parameters watchdog and nonmonotone line search redefined later
m=10       # default: 10
kwatch=20  # default: 20
watchdog=1 # 1=watchdog strategy active, otherwise not

# parameters for preprocessor
preprocess=True  # 1=preprocessor used, otherwise not
presteps=20    # maximum number of preprocessing steps, default: 20

# trust-region parameters for preprocessor
delta=5        # default: 5
deltamin=1     # default: 1
deltamax=1e+25 # default: 1e+25
rho1=1e-4      # default: rho1=1e-4
rho2=0.75      # default: rho2=0.75
sigma1=0.5     # default: 0.5
sigma2=2       # default: 2
eta=0.95       # default: 0.95


# initializations
defaults = dict(
    eps1 = 1e-10,    # default: 1e-6
    eps2 = 1e-14,   # default: 1e-10
    null = 1e-8,    # default: 1e-8
    Big  = 1e10,    # default: 1e+10

    preprocess=True,  # 1=preprocessor used, otherwise not
    presteps=20   # maximum number of preprocessing steps, default: 20
)


def lmmcp(fun, Dfun, x0, lb, ub, verbose=True, options={}):


    # TODO: infinite bounds should be replaced by big value



    opts = dict(defaults)
    opts.update(options)

    eps1 = opts['eps1']
    eps2 = opts['eps2']
    null = opts['null']
    Big  = opts['Big']
    preprocess = opts['preprocess']
    #verbose = opts['verbose']

    x = x0
    delta = 5


    k=0

    # compute a feasible starting point by projection
    x = np.maximum(lb,np.minimum(x,ub))

    n = len(x)

    #print 'Name and dimension of the testproblem: {} {}\n'.format( name, n)

    Indexset = np.zeros((n,1))
    #I_l=find(lb>-Big & ub>Big)
    #I_u=find(lb<-Big & ub<Big)
    #I_lu=find(lb>-Big & ub<Big)
    I_l=(lb>-Big) & (ub>Big)  # not exactly the same definition
    I_u=(lb<-Big) & (ub<Big)
    I_lu=(lb>-Big) & (ub<Big)


    Indexset[I_l]=1
    Indexset[I_u]=2
    Indexset[I_lu]=3

    #I_f=find(Indexset==0)
    I_f = - (I_l | I_u | I_lu)


    # function evaluations
    Fx  = fun(x)
    DFx = Dfun(x)

    # choice of NCP-function and corresponding evaluations
    #from Phi3MCPPFB import Phi3MCPPFB as Phi
    #from DPhi3MCPPFB import DPhi3MCPPFB as DPhi
    Phi=Phi3MCPPFB
    DPhi=DPhi3MCPPFB

    Phix=Phi(x,Fx,lb,ub,lambda1,lambda2,n,Indexset)

    from numpy.linalg import norm
    normPhix=norm(Phix)
    Psix=0.5*np.dot( Phix.T, Phix)

    DPhix=DPhi(x,Fx,DFx,lb,ub,lambda1,lambda2,n,Indexset)

    DPsix=np.dot(DPhix.T,Phix)
    
    normDPsix=norm(DPsix)

    # save initial values
    x0=x
    Fx0=Fx
    DFx0=DFx

    Phix0=Phix
    Psix0=Psix
    normPhix0=normPhix
    DPhix0=DPhix
    DPsix0=DPsix
    normDPsix0=normDPsix

    # watchdog strategy

    aux = np.zeros( (m,1) )
    aux[0]=Psix
    MaxPsi=Psix

    if watchdog==1:
        kbest=k
        xbest=x
        Phibest=Phix
        Psibest=Psix
        DPhibest=DPhix
        DPsibest=DPsix
        normDPsibest= normDPsix

    if verbose:

        headline = '|{0:^5} | {1:10} | {2:12} | {3:10} |'.format( 'k',' Psi(x)', '||DPsi(x)||','stepsize' )
        stars = '-'*len(headline)
        print(stars)
        print(headline)
        print(stars)
#        print('   k               Psi(x)                || DPsi(x) ||    stepsize\n')
#        print('====================================================================')
        s = '|{:^' + str(len(stars)-2) + '}|'
        print(s.format('Output at starting point'))
        print(stars)
        print('|{0:5} | {1:10.3e} | {2:12.3f} |'.format( k,Psix, normDPsix) )


    import numpy.linalg
    if preprocess==1:
        if verbose:
            s = '|{:^' + str(len(stars)-2) + '}|'
            print(stars)
            print(s.format('Preprocessor'))
            print(stars)

        normpLM=1
        while (k < presteps) & (Psix > eps2) & (normpLM>null):

            k=k+1

          # choice of Levenberg-Marquardt parameter, note that we do not use
          # the condition estimator for large-scale problems, although this
          # may cause numerical problems in some examples

            i=0
            mu=0
            if n<100:
                i=1
                mu=1e-16
                if numpy.linalg.cond(np.dot(DPhix.T,DPhix))>1e25: #TODO use random estimator
                    mu=1e-6/(k+1)

            if i==1:
                A1 = np.row_stack([ DPhix , np.sqrt(mu)*np.eye(n) ] )
                A2 = np.concatenate([ -Phix, np.zeros(n) ])
    #            pLM = A1 \ A2
                pLM = np.linalg.lstsq(A1,A2)[0]
            else:
                # pLM = A1 \ A2
                pLM = -np.linalg.lstsq(DPhix,Phix)[0]

            normpLM=norm(pLM)

            pLM = pLM.flatten()

          # compute the projected Levenberg-Marquard step onto box Xk
            lbnew=np.maximum(np.minimum(lb-x,0),-delta)
            ubnew=np.minimum(np.maximum(ub-x,0),delta)
            d=np.maximum(lbnew,np.minimum(pLM,ubnew))
            xnew=x+d

          # function evaluations etc.
            Fxnew  = fun(xnew)
            DFxnew = Dfun(xnew)
            Phixnew= Phi(xnew,Fxnew,lb,ub,lambda1,lambda2,n,Indexset)
            Psixnew=0.5*np.dot(Phixnew.T,Phixnew)
            normPhixnew=norm(Phixnew)

         # update of delta
            if normPhixnew<=eta*normPhix:
                delta=max(deltamin,sigma2*delta)
            elif normPhixnew>5*eta*normPhix:
                delta=max(deltamin,sigma1*delta)

          # update
            x=xnew
            Fx=Fxnew
            DFx=DFxnew
            Phix=Phixnew
            Psix=Psixnew
            normPhix=normPhixnew
            DPhix=DPhi(x,Fx,DFx,lb,ub,lambda1,lambda2,n,Indexset)
            DPsix=np.dot(DPhix.T,Phix)
            normDPsix=norm(DPsix,ord=np.inf)

          # output at each iteration
            t=1
            if verbose:
                print('|{0:5} | {1:10.3e} | {2:12.3f} | {3:10.3f} |'.format( k, Psix, normDPsix, t) )

#                print('{}\t{}\t{}\t{}\n'.format(k,Psix,normDPsix,t))


    # terminate program or redefine current iterate as original initial point
    if (preprocess==1) & (Psix<eps2):
        if verbose:
            s = '|{:^' + str(len(stars)-2) + '}|'
            print(stars)
            print(s.format('Approximate solution found'))
            print(stars)
            return x

    elif (preprocess==1) & (Psix>=eps2):
        k_main=0

        x=x0
        Fx=Fx0
        Phix=Phix0
        Psix=Psix0
        normPhix=normPhix0
        DPhix=DPhix0
        DPsix=DPsix0
        normDPsi=normDPsix0
        if verbose:
            s = '|{:^' + str(len(stars)-2) + '}|'
            print(s.format('Restart with initial point'))
            print('{}\t{}\t{}\n'.format(k_main,Psix0,normDPsix0))



    #
    #   Main algorithm
    #
    if verbose:
        s = '|{:^' + str(len(stars)-2) + '}|'
        print(stars)
        print(s.format('Main program'))
        print(stars)

    k_main = 0

    while (k < kmax) & (Psix > eps2):

       # choice of Levenberg-Marquardt parameter, note that we do not use
       # the condition estimator for large-scale problems, although this
       # may cause numerical problems in some examples

        i=0
        if n<100:
            i=1
            mu=1e-16
            if numpy.linalg.cond(np.dot(DPhix.T,DPhix))>1e25: #TODO use random estimator
                mu=1e-1/(k+1)
        if i==1:
            A1 = np.row_stack([ DPhix , np.sqrt(mu)*np.eye(n) ] )
            A2 = np.concatenate([ -Phix, np.zeros(n) ])
    #            pLM = A1 \ A2
            d = np.linalg.lstsq(A1,A2)[0]
        else:
            # pLM = A1 \ A2
            d = -np.linalg.lstsq(DPhix,Phix)[0]

        d = d.flatten()



         # computation of steplength t using the nonmonotone Armijo-rule
           # starting with the 6-th iteration

           # computation of steplength t using the monotone Armijo-rule if
           # d is a 'good' descent direction or k<=5

        t       = 1
        xnew    = x+d
        Fxnew   = fun(xnew)
        Phixnew = Phi(xnew,Fxnew,lb,ub,lambda1,lambda2,n,Indexset)
        Psixnew = 0.5*np.dot(Phixnew.T,Phixnew)
        const   = np.dot(sigma*DPsix.T,d)

        while (Psixnew > MaxPsi + const*t)  & (t > tmin):
            t       = t*beta
            xnew    = x+t*d
            Fxnew   = fun(xnew)
            Phixnew = Phi(xnew,Fxnew,lb,ub,lambda1,lambda2,n,Indexset)
            Psixnew = 0.5*np.dot(Phixnew.T,Phixnew)

        # updatings
        x=xnew
        Fx=Fxnew
        Phix=Phixnew
        Psix=Psixnew
        DFx=Dfun(x)
        DPhix=DPhi(x,Fx,DFx,lb,ub,lambda1,lambda2,n,Indexset)
        DPsix=np.dot(DPhix.T,Phix)
        normDPsix=norm(DPsix)
        k=k+1

        k_main=k_main+1

        if k_main<=5:
            aux[numpy.mod(k_main,m)]=Psix
            MaxPsi=Psix
        else:
            aux[numpy.mod(k_main,m)]=Psix
            MaxPsi=max(aux)

        # updatings for the watchdog strategy
        if watchdog ==1:
            if Psix<Psibest:
                 kbest=k
                 xbest=x
                 Phibest=Phix
                 Psibest=Psix
                 DPhibest=DPhix
                 DPsibest=DPsix
                 normDPsibest= normDPsix
            elif k-kbest>kwatch:
                 x=xbest
                 Phix=Phibest
                 Psix=Psibest
                 DPhix=DPhibest
                 DPsix=DPsibest
                 normDPsix=normDPsibest
                 MaxPsi=Psix
       # output at each iteration
        if verbose:
            print('{}\t{}\t{}\t{}\n'.format(k,Psix,normDPsix,t))

    if k == kmax:
        raise Exception('No convergence')
    return x


################### helper functions #######################

def Phi3MCPPFB(x,Fx,lb,ub,lambda1,lambda2, n,Indexset):
    y = np.zeros( 2*n );
    for i in range(1,n+1):
        phi_u = np.sqrt( (ub[i-1]-x[i-1])**2+Fx[i-1]**2) - ub[i-1] + x[i-1] +  Fx[i-1] ;
        if Indexset[i-1]==1:
            y[i-1] = lambda1*(-x[i-1]+lb[i-1] -Fx[i-1] +np.sqrt((x[i-1]-lb[i-1])**2 +Fx[i-1]**2));
            y[n+i-1] = lambda2*max(0,x[i-1]-lb[i-1])*max(0,Fx[i-1]);
        elif Indexset[i-1]==2:
            y[i-1]=-lambda1*phi_u;
            y[n+i-1]= lambda2*max(0,ub[i-1]-x[i-1])*max(0,-Fx[i-1]);
        elif Indexset[i-1]==0:
            y[i-1]=-lambda1*Fx[i-1];
            y[n+i-1]= -lambda2*Fx[i-1];
        elif Indexset[i-1]==3:
            y[i-1]=lambda1*(np.sqrt((x[i-1]-lb[i-1])**2+phi_u**2)-x[i-1]+lb[i-1]-phi_u);
            y[n+i-1]=lambda2*(max(0,x[i-1]-lb[i-1])*max(0,Fx[i-1])+max(0,ub[i-1]-x[i-1])*max(0,-Fx[i-1]));
    return y


def DPhi3MCPPFB(x, Fx, DFx, lb, ub, lambda1, lambda2, n, Indexset):

    #% we evaluate an element of the C-subdifferential of operator Phi3MCPPFB
    null = 1e-8
    beta_l = np.zeros( n )
    beta_u = np.zeros( n )
    alpha_l = np.zeros( n )
    alpha_u = np.zeros( n )
    z = np.zeros( n )
    H1 = np.zeros( (n, n) ) # should be sparse
    H2 = H1.copy()
    for i in range(1, n+1):
        if np.logical_and(np.abs((x[i-1]-lb[i-1]))<=null, np.abs(Fx[i-1])<=null):
            beta_l[i-1] = 1.
            z[i-1] = 1.


        if np.logical_and(np.abs((ub[i-1]-x[i-1]))<=null, np.abs(Fx[i-1])<=null):
            beta_u[i-1] = 1.
            z[i-1] = 1.


        if np.logical_and(x[i-1]-lb[i-1] >= -null, Fx[i-1] >= -null):
            alpha_l[i-1] = 1.


        if np.logical_and(ub[i-1]-x[i-1] >= -null, Fx[i-1]<=null):
            alpha_u[i-1] = 1.



    Da = np.zeros( (n, 1) )
    Db = np.zeros( (n, 1) )
    for i in range(1, n+1):
#        ei = np.zeros( (1., n) )
        ei = np.zeros( n )
        ei[i-1] = 1.
        if Indexset[i-1] == 0.:
            Da[i-1] = 0
            Db[i-1] = -1
            H2[i-1,:] = -DFx[i-1,:]
        elif Indexset[i-1] == 1.:
            # TODO : hyp : the maximand is a scalar
            denom1 = np.maximum(null, np.sqrt(((x[i-1]-lb[i-1])**2.+Fx[i-1]**2.)))
            denom2 = np.maximum(null, np.sqrt((z[i-1]**2.+np.dot(DFx[i-1,:], z)**2.)))
            if beta_l[i-1] == 0.:
                Da[i-1] = (x[i-1]-lb[i-1])/ denom1-1.
                Db[i-1] = Fx[i-1] / denom1-1.
            else:
                Da[i-1] = z[i-1] / denom2 - 1.
                Db[i-1] = np.dot(DFx[i-1,:], z) / denom2 -1.


            if alpha_l[i-1] == 1.:
                H2[i-1,:] = (x[i-1]-lb[i-1]) * DFx[i-1,:] + Fx[i-1] * ei
            else:
                H2[i-1,:] = 0.



        elif Indexset[i-1] == 2.:
            denom1 = np.maximum(null, np.sqrt(((ub[i-1]-x[i-1])**2.+Fx[i-1]**2.)))
            denom2 = np.maximum(null, np.sqrt((z[i-1]**2.+np.dot(DFx[i-1,:], z)**2.)))
            if beta_u[i-1] == 0.:
                Da[i-1] = (ub[i-1]-x[i-1]) / denom1-1.
                Db[i-1] = -Fx[i-1] / denom1-1.
            else:
                Da[i-1] = -z[i-1]/ denom2 - 1.
                Db[i-1] = -np.dot(DFx[i-1,:], z) / denom2 -1.


            if alpha_u[i-1] == 1.:
                H2[i-1,:] = np.dot(x[i-1]-ub[i-1], DFx[i-1,:])+np.dot(Fx[i-1], ei)
            else:
                H2[i-1,:] = 0.



        elif Indexset[i-1] == 3.:
            ai = 0.
            bi = 0.
            ci = 0.
            di = 0.
            phi = -ub[i-1]+x[i-1]+Fx[i-1]+np.sqrt(((ub[i-1]-x[i-1])**2.+Fx[i-1]**2.))
            denom1 = np.maximum(null, np.sqrt(((x[i-1]-lb[i-1])**2.+phi**2.)))
            denom2 = np.maximum(null, np.sqrt((z[i-1]**2.+np.dot(DFx[i-1,:], z)**2.)))
            denom3 = np.maximum(null, np.sqrt(((ub[i-1]-x[i-1])**2.+Fx[i-1]**2.)))
            denom4 = np.maximum(null, np.sqrt((z[i-1]**2.+(np.dot(ci, z[i-1])+np.dot(np.dot(di, DFx[i-1,:]), z))**2.)))

            if beta_u[i-1] == 0.:
                ci = (x[i-1]-ub[i-1]) / denom3 + 1
                di = Fx[i-1] / denom3 + 1
            else:
                ci = 1 + z[i-1] / denom2
                di = 1 + np.dot(DFx[i-1,:], z)/ denom2


            if beta_l[i-1] == 0.:
                ai = (x[i-1]-lb[i-1]) / denom1 - 1
                bi = phi / denom1 - 1
            else:
                ai = z[i-1] / denom4 - 1.
                bi = ( np.dot(ci, z[i-1]) + np.dot(np.dot(di, DFx[i-1,:]), z) ) / denom4-1.


            Da[i-1] = ai+np.dot(bi, ci)
            Db[i-1] = np.dot(bi, di)
            if np.logical_and(alpha_l[i-1] == 1., alpha_u[i-1] == 1.):
                H2[i-1,:] = np.dot(-lb[i-1]-ub[i-1]+2.*x[i-1], DFx[i-1,:])+np.dot(2.*Fx[i-1], ei)
            else:
                if alpha_l[i-1] == 1.:
                    H2[i-1,:] = np.dot(x[i-1]-lb[i-1], DFx[i-1,:])+np.dot(Fx[i-1], ei)
                elif alpha_u[i-1] == 1.:
                    H2[i-1,:] = np.dot(x[i-1]-ub[i-1], DFx[i-1,:])+np.dot(Fx[i-1], ei)

                else:
                    H2[i-1,:] = 0.


        H1[i-1,:] = Da[i-1] * ei + Db[i-1] * DFx[i-1,:]

#  H=[lambda1*H1; lambda2*H2];
    H=np.row_stack([lambda1*H1, lambda2*H2]);
#    H = np.array(np.vstack((np.hstack((np.dot(lambda1, H1))), np.hstack((np.dot(lambda2, H2))))))

    return H