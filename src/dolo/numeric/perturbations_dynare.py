# Here we use Dynare to compute the perturbations


    



def retrieve_DDR_from_matlab(name,mlab):
    mlab.execute( 'drn = reorder_dr({0});'.format(name) )
    rdr = retrieve_from_matlab('drn',mlab)
    ys =  rdr['ys'][0,0].flatten()
    ghx = rdr['ghx'][0,0]
    ghu = rdr['ghu'][0,0]
    [n_v,n_states] = ghx.shape
    n_shocks = ghu.shape[1]
    ghxx = rdr['ghxx'][0,0].reshape( (n_v,n_states,n_states) )
    ghxu = rdr['ghxu'][0,0].reshape( (n_v,n_states,n_shocks) )
    ghuu = rdr['ghuu'][0,0].reshape( (n_v,n_shocks,n_shocks) )
    ghs2 = rdr['ghs2'][0,0].flatten()
    ddr = DDR( [ ys,[ghx,ghu],[ghxx,ghxu,ghuu] ] , ghs2 = ghs2 )
    return [ddr,rdr]


dr1_info_code = {
1: "the model doesn't define current variables uniquely.",
2: "problem in mjdgges.dll info(2) contains error code.",
3: "BK order condition not satisfied info(2) contains 'distance' absence of stable trajectory.",
4: "BK order condition not satisfied info(2) contains 'distance' indeterminacy.",
5: "BK rank condition not satisfied.",
6: "The jacobian matrix evaluated at the steady state is complex."
}
