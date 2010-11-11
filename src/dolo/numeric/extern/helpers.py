'''(Mostly time-series-related) functions needed and written by Sven Schreiber.

This is free but copyrighted software, distributed under the same license terms
(as of January 2007) as the 'gretl' program by Allin Cottrell and others, see
gretl.sf.net (in short: GPL v2, see www.gnu.org/copyleft/gpl.html).

(see end of this file for a changelog)
'''
from numpy import r_, c_, arange, diff, mean, sqrt, log, mat
from numpy import asarray, nan
from numpy.matlib import ones, zeros, rand, eye, empty
from numpy.linalg import eigh, cholesky, solve, lstsq
# (lstsq also as tool to determine rank)

# some constants/dictionaries first
quarter2month = {1: 1, 2: 4, 3: 7, 4: 10}
# in theory we only need the four months 1, 4, 7, 10, but well...
month2quarter = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, \
                 10: 4, 11: 4, 12: 4}
qNumber2qFloat = {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75}
mNumber2mFloat = {1: 0.0, 2: 0.0833, 3: 0.1666, 4: 0.2499, 5: 0.3332, \
                  6: 0.4165, 7: 0.4998, 8: 0.5831, 9: 0.6664, 10: 0.7497, \
                  11: 0.8330, 12: 0.9163}
qFracstring2qString = {0.0: 1, 0.25: 2, 0.5: 3, 0.75: 4}
mFloat2mNumber = {0.0: 1, 0.0833: 2, 0.1666: 3, 0.2499: 4, 0.3332: 5, \
                  0.4165: 6, 0.4998: 7, 0.5831: 8, 0.6664: 9, 0.7497: 10, \
                  0.8330: 11, 0.9163: 12}
# with onetwelfth == 0.0833 approx.

from numpy.linalg import lstsq, svd
from numpy import where
def rank(m, rcond = 1e-10):
    '''
    Returns the (algebraic, not numpy-jargon) rank of m.
    '''
    svals = svd(m)[1]
    return where(svals > svals[0]*rcond, 1, 0).sum()

def vec(m):
    '''
    Returns all columns of the input as a stacked (column) vector.

    If m is a numpy-array, a 1d-array is returned. For a numpy-matrix m,
     the output has shape (n*m, 1).
    '''
    return m.T.ravel().T

from numpy import mat, asarray
def unvec(m, rows, cols):
    '''
    Turns (column) vector into matrix of shape == (rows, cols).

    Also accepts 1d-array input, but always returns numpy matrix.
    '''
    if type(m) == type(mat(m)):
        assert m.shape[1] == 1                            # col vector
        intype = 'matrix'
    else:
        assert len(m.shape) == 1                          # 1d array
        intype = 'array'
        m = mat(m).T
    assert cols * rows == m.shape[0]
    out = m.reshape(cols, rows).T
    if intype == 'array': return asarray(out)
    else: return out

from numpy import mat
def mat2gretlmatstring(m):
    '''
    Turns numpy matrix or array (or scalar!) m into gretl string representation.
    '''
    # mat(m) is necessary because if m is 1d-array, map() would fail
    out =  ';'.join( [  ','.join(map(str, row)) for row in mat(m).tolist() ] )
    return '{' + out + '}'

def startobs2obslist(startperiod, numofobs):
    '''
    Constructs list of observation labels following the input pattern.

    Example:
    startperiod = '1999q3', numofobs = 2 -> ['1999q3', '1999q4']
    Currently supports only annual (pure number), monthly, quarterly.
    Years must be in 4-digit format.
    '''
    if startperiod.isdigit():           # pure (integer) number
        startnumber = int(startperiod)
        return [ str(startnumber + ix) for ix in range(numofobs) ]
    elif startperiod[4] in 'qQ':        # quarterly dates
        wrap = 4
        period = int(startperiod[5])
    elif startperiod[4] in 'mM':
        wrap = 12
        period = int(startperiod[5:7])
    else: raise NotImplementedError

    year = int(startperiod[:4])
    out = [str(year) + startperiod[4] + str(period)]
    for ix in range(numofobs):
        if period == wrap:
            period = 1
            year += 1
        else: period += 1
        out.append(str(year) + startperiod[4] + str(period))

    return out

import csv
from numpy import mat
def writecsv(filename, data, orientation = 'cols', delim = ',', \
     varnames = [],  obslabels = [], comments = [], commentchar = '# '):
    '''
    Saves array or matrix <data> in csv format in file <filename> (path string).

    <comments> must be passed as a sequence of strings, one for each line,
     and will be written at the top of the file, each line starting with 
     <commentchar>.
    <orientation> can be 'cols' or 'rows', determines whether the
     variable names will be used as column or row headers, and how to treat
     1d-input. (And observation labels will be written accordingly.)
    <varnames> and <obslabels> must be sequences of strings.
    '''
    data = mat(data)
    if orientation == 'rows':
        colheaders = obslabels
        rowheaders = varnames
        cell11 = 'var'
    else:                           # 'cols' orientation as fallback
        colheaders = varnames
        rowheaders = obslabels
        cell11 = 'obs'
        if data.shape[0] == 1: data = data.T    # make 1d-array a column vector
    if len(colheaders) > 0: assert len(colheaders) == data.shape[1]

    # start writing to the file
    target = csv.writer(open(filename, 'w'), delimiter = delim)
    target.writerows([ [commentchar + comment] for comment in comments])
    # (additional trivial list layer because otherwise the comment string itself
    #  would be split up with the delim character)
    if len(rowheaders) > 0:
        assert len(rowheaders) == data.shape[0]
        target.writerow(colheaders.insert(0, cell11))
    else: target.writerow(colheaders)
    temp = data.tolist()        # temp to have list conversion only once
    for ix in range(len(rowheaders)): temp[ix].insert(0, rowheaders[ix])
    target.writerows(temp)

    return 0            # success

import csv
from numpy import mat
def readcsv(filename, delim = ',', commentchar = '#', colheader = 'names', \
        rowheader = 'obs'):
    '''
    Read in a csv file (may contain comments starting with commentchar).

    The contents of the first non-comment row and column must be indicated in
     rowheader and colheader as one of 'names', 'obs' (labels), or None.
    The array (matrix) of the data is returned as is, i.e. w/o transpose, hence
     the caller must know whether variables are in rows or columns.
    If both colheader and rowheader are not None, the upper-left cell (header
     of the first row/col) is ignored (but must be non-empty).

    Returns a five-element tuple:
    0. numpy-matrix of the actual data as floats
    1. orientation of variables: 'cols', 'rows', or 'unknown'
    2. 1d-array of variable names (or None)
    3. 1d-array of observation labels (or None)
    4. the type/frequency of the data
        (currently one of 'a', 'q', 'm', guessed from the first date label)
        (if this deduction failed, 'unknown' is returned here)

    Easiest example with upper-left data cell in second row/second column:
    mydata = readcsv('myfile.csv')[0]
    '''
    read_from = csv.reader(open(filename, 'rb'), delimiter = delim, \
        skipinitialspace = True)
    tempnestedlist = [ line for line in read_from if not \
        line[0].strip().startswith(commentchar) ]
    data = mat(tempnestedlist, dtype = str)

    if colheader == 'names':
        orientation = 'cols'
        varnames, data = data[0, :].A1, data[1:, :]
        if rowheader == 'obs':
            obslabels, data = data[:, 0].A1, data[:, 1:]
            varnames = varnames[1:]
    elif rowheader == 'names':
        orientation = 'rows'
        varnames, data = data[:, 0].A1, data[:, 1:]
        if colheader == 'obs':
            obslabels, data = data[0, :].A1, data[1:, :]
            varnames = varnames[1:]
    elif colheader == 'obs':
        orientation = 'rows'
        obslabels, data = data[0, :].A1, data[1:, :]
        if rowheader == 'names':
            varnames, data = data[:, 0].A1, data[:, 1:]
            obslabels = obslabels[1:]
    elif rowheader == 'obs':
        orientation = 'cols'
        obslabels, data = data[:, 0].A1, data[:, 1:]
        if colheader == 'names':
            varnames, data = data[0, :].A1, data[1:, :]
            obslabels = obslabels[1:]
    else:
        assert colheader == None        # to catch typos, e.g. 'Names', 'OBS'
        assert rowheader == None
        orientation = 'unknown'
        varnames = None
        obslabels = None

    # detect the dataset type:
    # annual
    if len(obslabels[0]) == 4: freq = 'a'
    # quarterly
    elif len(obslabels[0]) == 6 and obslabels[0][4] in 'qQ': freq = 'q'
    # monthly
    elif len(obslabels[0]) == 7 and obslabels[0][4] in 'mM': freq = 'm'
    else: freq = 'unknown'

    return data.astype(float), orientation, varnames, obslabels, freq

from numpy import nan
def floatAndNanConverter(datapoint, nacode = 'na'):
    '''
    Converts nacode to numpy.nan value.

    Also returns other input as float (e.g. for matplotlib's load, asarray).
    '''
    if datapoint == nacode: return nan
    return float(datapoint)

def dateString2dateFloat(datestring):
    '''
    Converts '1999q2' -> 1999.25, '1999m2' -> 1999.0833, etc.

    So far only for quarterly and monthly.
    '''
    year, freq = float(datestring[:4]), datestring[4]
    assert freq in 'qQmM', 'sorry, only quarterly or monthly'
    if freq in 'qQ':    #quarterly
        result = year + qNumber2qFloat[int(datestring[5])]
    elif freq in 'mM':               #monthly
        result = year + mNumber2mFloat[int(datestring[5:7])]
    return result

from datetime import date, timedelta
def getQuarterlyDates(startyear, startquarter, t):
    '''
    Constructs a list of quarterly date labels for t obs.

    Algorithm to get a sequence of strings relating to quarterly dates:
     1. start with first day in the startquarter, e.g. 2006-04-01
     2. map the month to quarter and make string year + 'q' + quarter
     3. the longest quarters are 3rd and 4th (2*31 days + 30 days = 92 days),
        1st the shortest (90 or 91), so add a timedelta (in days,
        apparently default) of 100 days (anything between 92+1 and
        sum of shortest quarter plus one month = approx. 118)
     4. reset the day of that intermediate date to 1
     5. return to step 2
    '''
    try:
        y = int(startyear); q = int(startquarter); t = int(t)
    except: raise TypeError, 'need integers for year, quarter, t'
    if q not in range(1,5): raise ValueError, 'startquarter input out of range'
    # create list for date strings:
    datestrings = []
    # step 1.:
    d = date(y, quarter2month[startquarter], 1)
    for t in range(t):
        datestrings.append(str(d.year) + 'Q' + str(month2quarter[d.month]))
        d += timedelta(100)
        d = d.replace(day = 1)
    return datestrings

from numpy.linalg import svd
def null(m, rcond = 1e-10):
    rows, cols = m.shape
    u, svals, vh = svd(m)
    rk = where(svals > svals[0]*rcond, 1, 0).sum()
    return u[:, rk:]

from numpy.matlib import empty, zeros, eye, mat, asarray
from numpy.linalg import lstsq
def getOrthColumns(m):
    '''
    Constructs the orthogonally complementing columns of the input.

    Input of the form pxr is assumed to have r<=p,
    and have either full column rank r or rank 0 (scalar or matrix)
    Output is of the form px(p-r), except:
    a) if M square and full rank p, returns scalar 0
    b) if rank(M)=0 (zero matrix), returns I_p
    (Note you cannot pass scalar zero, because dimension info would be
    missing.)
    Return type is as input type.
    '''
    if type(m) == type(asarray(m)):
        m = mat(m)
        output = 'array'
    else: output = 'matrix'
    p, r = m.shape
    # first catch the stupid input case
    if p < r: raise ValueError, 'need at least as many rows as columns'
    # we use lstsq(M, ones) just to exploit its rank-finding algorithm,
    rk = lstsq(m, ones(p).T)[2]
    # first the square and full rank case:
    if rk == p: result = zeros((p,0))   # note the shape! hopefully octave-like
    # then the zero-matrix case (within machine precision):
    elif rk == 0: result = eye(p)
    # now the rank-deficient case:
    elif rk < r:
        raise ValueError, 'sorry, matrix does not have full column rank'
    # (what's left should be ok)
    else:
        # we have to watch out for zero rows in M,
        # if they are in the first p-r positions!
        # so the (probably inefficient) algorithm:
            # 1. check the rank of each row
            # 2. if zero, then also put a zero row in c
            # 3. if not, put the next unit vector in c-row
        idr = eye(r)
        idpr = eye(p-r)
        c = empty([0,r])    # starting point  
        co = empty([0, p-r]) # will hold orth-compl.
        idrcount = 0
        for row in range(p):
            # (must be ones() instead of 1 because of 2d-requirement
            if lstsq( m[row,:], ones(1) )[2] == 0 or idrcount >= r:
                c = r_[ c, zeros(r) ]
                co = r_[ co, idpr[row-idrcount, :] ]
            else:     # row is non-zero, and we haven't used all unit vecs 
                c = r_[ c, idr[idrcount, :] ] 
                co = r_[ co, zeros(p-r) ]
                idrcount += 1
        # earlier non-general (=bug) line: c = mat(r_[eye(r), zeros((p-r, r))])
        # and:  co = mat( r_[zeros((r, p-r)), eye(p-r)] )
        # old:
        # result = ( eye(p) - c * (M.T * c).I * M.T ) * co
        result = co - c * solve(m.T * c, m.T * co)
    if output == 'array': return result.A
    else: return result

from numpy import mat, asarray
def addLags(m, maxlag):
    '''
    Adds (contiguous) lags as additional columns to the TxN input.

    Early periods first. If maxlag is zero, original input is returned.
    maxlag rows are deleted (the matrix is shortened) 
    '''
    if type(m) == type(asarray(m)): 
        m = mat(m)
        output = 'array'
    else: output = 'matrix'
    T, N = m.shape
    if type(maxlag) != type(4):
        raise TypeError, 'addLags: need integer for lag order'
    if maxlag > m.shape[0]:
        raise ValueError, 'addLags: sample too short for this lag'
    temp = m[ maxlag: ,:]  # first maxlag periods must be dropped due to lags
    for lag in range(1, maxlag + 1) :
        temp = c_[ temp, m[(maxlag-lag):(T-lag) ,:] ]
    if output == 'array': return asarray(temp)
    else: return temp

from numpy.matlib import empty, ones, zeros
from numpy import mat, c_, r_
def getDeterministics(nobs, which = 'c', date = 0.5):
    '''
    Returns various useful deterministic terms for a given sample length T.

    Return object is a numpy-matrix-type of dimension Tx(len(which));
    (early periods first, where relevant).
    In the 'which' argument pass a string composed of the following letters,
    in arbitrary order:
    c - constant (=1) term
    t - trend (starting with 0)
    q - centered quarterly seasonal dummies (starting with 0.75, -0.25...)
    m - centered monthly seasonal dummies (starting with 11/12, -1/12, ...)
    l - level shift (date applies) 
    s - slope shift (date applies) 
    i - impulse dummy (date applies)

    If the date argument is a floating point number (between 0 and 1),
    it is treated as the fraction of the sample where the break occurs.
    If instead it is an integer between 0 and T, then that observation is
    treated as the shift date.    
    '''
    # some input checks (as well as assignment of shiftperiod):
    if type(nobs) != type(4):  # is not an integer
        raise TypeError, 'need integer for sample length'
    if nobs <=0: raise ValueError, 'need positive sample length'
    if type(date) == type(0.5):     #is a float, treat as break fraction
        if date < 0 or date > 1:
            raise ValueError, 'need break fraction between 0 and 1'
        shiftperiod = int(date * nobs)
    elif type(date) == type(4):     # is integer, treat as period number
        if date not in range(1, nobs+1):
            raise ValueError, 'need period within sample range'
        shiftperiod = date
    else: raise TypeError, 'need float or integer input for date'
    if type(which) != type('a string'):
        raise TypeError, 'need string for case spec' 
    # end input checks

    out = empty([nobs,0])   # create starting point
    if 'c' in which: out = c_[ out, ones(nobs).T ]
    if 't' in which: out = c_[ out, r_['c', :nobs] ]
    if 'l' in which:
        shift = r_[ zeros(shiftperiod).T, ones(nobs-shiftperiod).T ]
        out = c_[ out, shift ]
    if 's' in which:
        slopeshift = r_[ zeros(shiftperiod).T, r_['c', 1:(nobs - shiftperiod + 1)] ]
        out = c_[ out, slopeshift ]
    if 'i' in which:
        impulse = r_[ zeros(shiftperiod).T, ones(1), zeros(nobs-shiftperiod-1).T ]
        out = c_[ out, impulse ]
    if 'q' in which or 'Q' in which:
        # to end of next full year, thus need to slice at T below:
        q1 = [0.75, -0.25, -0.25, -0.25] * (1 + nobs/4)
        q2 = [-0.25, 0.75, -0.25, -0.25] * (1 + nobs/4)
        q3 = [-0.25, -0.25, 0.75, -0.25] * (1 + nobs/4)
        out = c_[ out, mat(q1[:nobs]).T, mat(q2[:nobs]).T, mat(q3[:nobs]).T ]
    if 'm' in which or 'M' in which:
        temp = [-1./12] * 11 
        for month in range(11):
            temp.insert(month, 1-temp[0])
            # again, to end of next full year, thus need to slice at T below:
            monthly = temp * (1 + nobs/12)  # temp is still a list here!
            out = c_[ out, mat(monthly[:nobs]).T ]
    return out

from numpy.matlib import empty
def getImpulseDummies(sampledateslist, periodslist):
    '''
    Returns a (numpy-)matrix of impulse dummies for the specified periods.

    sampledateslist must consist of 1999.25 -style dates (quarterly or monthly).
    However, because periodslist is probably human-made, it expects strings
     such as '1999q3' or '1999M12'.
    Variables in columns.
    So far only for quarterly and monthly data.
    '''
    nobs = len(sampledateslist)
    result = empty([nobs,0])
    for periodstring in periodslist:
        period = dateString2dateFloat(periodstring)
        result = c_[result, getDeterministics(nobs, 'i', \
                            sampledateslist.index(period))]
    return result

from numpy import mat, asarray
from numpy.linalg import cholesky, eigh
def geneigsympos(A, B):
    ''' Solves symmetric-positive-def. generalized eigenvalue problem Az=lBz.

    Takes two real-valued symmetric matrices A and B (B must also be
    positive-definite) and returns the corresponding (also real-valued)
    eigenvalues and eigenvectors. 

    Return format: as in scipy.linalg.eig, tuple (l, Z); l is taken from eigh
    output (a 1-dim array of length A.shape[0] ?) ordered ascending, and Z is
    an array or matrix (depending on type of input A) with the corresponding
    eigenvectors in columns (hopefully).

    Steps:
        1. get lower triang Choleski factor of B: L*L.T = B
         <=> A (LL^-1)' z = l LL' z
         <=> (L^-1 A L^-1') (L'z) = l (L'z)
        2. standard eig problem, with same eigvals l
        3. premultiply eigvecs L'z by L^-1' to get z
    '''
    output = 'matrix'
    if type(A) == type(asarray(A)):
        output = 'array'
        A, B = mat(A), mat(B)
    # step 1
    LI = cholesky(B).I
    # step 2
    evals, evecs = eigh(LI * A * LI.T)
    # sort
    evecs = evecs[:, evals.argsort()]
    evals.sort()        # in-place!
    # step 3
    evecs = LI.T * evecs
    if output == 'array': return evals, asarray(evecs) 
    else:   return evals, evecs

from numpy.matlib import eye, c_
def vecm2varcoeffs(gammas, maxlag, alpha, beta):
    '''
    Converts Vecm coeffs to levels VAR representation.

    Gammas need to be coeffs in shape #endo x (maxlag-1)*#endo,
    such that contemp_diff = alpha*ect + Gammas * lagged_diffs 
    is okay when contemp_diff is  #endo x 1.
    We expect matrix input!
    '''
    if alpha.shape != beta.shape:   # hope this computes for tuples
        raise ValueError, 'alpha and beta must have equal dim'
    N_y = alpha.shape[0]
    if beta.shape[0] != N_y:
        raise ValueError, "alpha or beta dim doesn't match"
    if gammas.shape[0] != N_y:
        raise ValueError, "alpha or gammas dim doesn't match"
    if gammas.shape[1] != (maxlag-1)*N_y:
        raise ValueError, "maxlag or gammas dim doesn't match"

    # starting point first lag:
    levelscoeffs = eye(N_y) + alpha * beta.T + gammas[ : , :N_y ]
    # intermediate lags:
    for lag in range(1, maxlag-1):
        levelscoeffs = c_[ levelscoeffs, gammas[:, N_y*lag : N_y*(lag+1)] - \
                          gammas[:,  N_y*(lag-1) : N_y*lag ] ]
    # last diff-lag, now this should be N_y x maxlags*N_y:
    return c_[ levelscoeffs, -gammas[:, -N_y: ] ]

def gammas2alternativegammas(gammas, alpha, beta):
    '''
    Converts Vecm-coeffs for ect at t-1 to the ones for ect at t-maxlag.

    The input gammas (shortrun coeffs) refer to a Vecm where the levels are 
     lagged one period. In the alternative representation with the levels 
     lagged maxlag periods the shortrun coeffs are different; the relation is:
         alt_gamma_i = alpha * beta' + gamma_i

    Actually with numpy's broadcasting the function is a one-liner so this here
     is mainly for documentation and reference purposes.
    In terms of the levels VAR coefficients A_i (i=1..maxlag) the gammas are
     defined as:
         gamma_i = - \sum_{j=i+1)^maxlag A_j for i=1..maxlag-1;
     and the alternative gammas (used py Proietti e.g.) are:
         alt_gamma_i = -I + \sum_{j=1}^i A_j for i=1..maxlag-1.
     (And \alpha \beta' = -I + \sum_{j=1}^maxlag A_j.)
    '''
    # use broadcasting to do the summation in one step:
    return  alpha * beta.T + gammas

import os
from numpy.matlib import mat
def write_gretl_mat_xml(outfile, matrices, matnames = []):
        '''
        Writes a gretl matrix xml file to transfer matrices.

        outfile should be a path string,
        matrices is a list of numpy matrices, 
        matnames is a string list of wanted matrix names (if empty, matrices
         are named m1, m2, etc.)
        '''
        if matnames == []:
            matnames = ['m' + str(mindex) for mindex in range(len(matrices))]
        assert len(matrices) == len(matnames)
        out = open(outfile, 'w')
        out.write('<?xml version="1.0" encoding="UTF-8"?>' + os.linesep)
        out.write('<gretl-matrices count="' + str(len(matrices)) + '">' + os.linesep)
        for m in matrices:
            out.write('<gretl-matrix name="' + matnames.pop(0) + '" ')
            out.write('rows="' + str(m.shape[0]) + '" ')
            out.write('cols="' + str(m.shape[1]) + '">' + os.linesep)
            for row in m: out.write(str(row).strip('][') + os.linesep)
            out.write('</gretl-matrix>' + os.linesep)
        out.write('</gretl-matrices>')
        out.close()

################################
## now some more econometrically oriented helper functions
################################

from numpy.matlib import zeros, mat, asarray
def autocovar(series, LagInput, Demeaned=False):
    '''
    Computes the autocovariance of a uni- or multivariate time series.

    Usage: autocovar(series, Lag [, Demeaned=False]) returns the NxN
    autocovariance matrix (even for N=1), where series is
    an TxN matrix holding the N-variable T-period data (early periods first),
    and Lag specifies the lag at which to compute the autocovariance.
    Specify Demeaned=True if passing ols-residuals to avoid double demeaning.
    Returns a numpy-matrix-type.
    '''
    if type(series) == type(asarray(series)): 
        output = 'array'
        series = mat(series)
    else: output = 'matrix'
    t, n = series.shape
    try: Lag = int(LagInput)
    except: raise TypeError, 'autocovar: nonsense lag input type'
    if Demeaned == False:
        # axis=0 for columns (otherwise does overall-average):
        xbar = series.mean(axis=0)
    else: xbar = 0              # seems to broadcast to vector-0 ok (below)
    result = zeros([n,n])
    for tindex in range(Lag, t):
        xdev1 = series[tindex,:] - xbar
        xdev2 = series[tindex-Lag, :] - xbar
        result += xdev1.T * xdev2
    result /= t
    if output == 'array': return asarray(result)
    else: return result

from numpy.matlib import zeros, mat, asarray
def longrunvar(series, Demeaned = False, LagTrunc = 4):
    '''
    Estimates the long-run variance (aka spectral density at frequency zero)
    of a uni- or multivariate time series.

    Usage: lrv = longrunvar(series [, Demeaned, LagTrunc]),
    where series is a TxN matrix holding
    the N-variable T-period data (early periods first).
    The Bartlett weighting function is used
    up to the specified lag truncation (default = 4).
    Specify Demeaned=True when passing Ols-residuals etc. (default False).
    Returns an NxN matrix (even for N=1).
    '''
    if type(series) == type(asarray(series)): 
        output = 'array'
        series = mat(series)
    else: output = 'matrix'
    t, n = series.shape

    # set the lag window constant:
    try: Lag = int(LagTrunc)
    except: raise TypeError, 'longrunvar: nonsense lag input type'
    if Lag >= t-1:
        Lag = int(sqrt(t))
        print 'longrunvar warning: not enough data for chosen lag window'
        print '(was ', LagTrunc, ', reset to ', Lag, ')'

    result = zeros([n,n])
    for tau in range(1, Lag+1):
        Gamma = autocovar(series, tau, Demeaned)    # numpy-matrix here
        #the positive and negative range together:
        result += (1-tau/(Lag+1)) * (Gamma + Gamma.T) 
    # add the tau=0 part:
    result +=  autocovar(series, 0, Demeaned)
    if output == 'array': return asarray(result)
    else: return result

from numpy.matlib import ones, zeros, mat
from numpy.linalg import solve
def commontrendstest(series, LagTrunc=4, determ = 'c', breakpoint=0.5):
    '''
    The Nyblom&Harvey(2000)-type tests for K_0 against K>K_0
    common stochastic trends in time series.

    Usage: 
    commontrendstest(series [, LagTrunc, Deterministics, breakpoint])
     returns a 1d N-array with the test statistics (partial sums of relevant
     eigenvalues), starting with the null hypothesis K_0=N-1 and ending with 
     K_0=0.
    
    Input:
    TxN array of data in series (early periods first).
    
    LagTrunc:
    determines the truncation lag of the nonparametric estimate of the
     longrun variance.
    
    Deterministics:
    'c' - constant mean,
    't' - to automatically de-trend the data (linearly),
    
    or use one of the following models with (one-time) deterministic shifts
    (see Busetti 2002):
    '1' - (a string!) for a level shift w/o trend,
    '2' - for a model with breaks in the mean and the trend slope,
    '2a' - for a trend model where only the mean shifts.
    (Case 2b --broken trends with connected segments-- is not implemented.)
    
    For these models '1' through '2a' the relative breakpoint in the sample can
     be chosen (otherwise it is ignored).
    '''
    series = mat(series)
    t, n = series.shape
    try: Lag = int(LagTrunc)
    except: raise TypeError, 'commontrendstest: nonsense lag input type'
    if Lag <= 0:
        print 'commontrendstest warning: lag trunc too small, set to default!'
        Lag = 4
    if type(breakpoint) != type(0.5):   # check for floating point input
        raise TypeError, 'commontrendstest: nonsense breakpoint input type'
    elif (breakpoint <= 0) or (breakpoint >= 1):
        raise ValueError, 'commontrendstest: breakpoint not in unit interval'
    
    if determ == 'c': D = ones(t).T   
    elif determ == 't': D = getDeterministics(t, 'ct')
    elif determ == '1': D = getDeterministics(t, 'cl', breakpoint)  
    elif determ == '2': D = getDeterministics(t, 'ctls', breakpoint)
    elif determ == '2a': D = getDeterministics(t, 'ctl', breakpoint)

    # okay, now remove the deterministics:
    # (by now, D should be Tx(1,2,3, or 4) )
    # this should do the projection:
    Resid = series - D * solve(D.T * D, D.T * series)
    Cmat = zeros([n,n])
    for i in range(t):
        temp = zeros((1,n))
        for tindex in range(i):
            temp += Resid[tindex,:]
        Cmat += temp.T * temp
    Cmat /= t**2
    Sm = longrunvar(Resid, True, Lag)
    # (True for data w/o deterministics, because everything removed)

    # generalized eigenvalues, corresponding to det(Cmat- lambda_j Sm)=0
    try: evals = geneigsympos(Cmat, Sm)[0]
    except:
        # most probably Sm wasn't pos-def, which can happen depending on lags,
        # then we try to switch over to scipy's more general eigenvalues
        print Sm    #to get some insight before everything dies
        try:
            from scipy import linalg as sl
            evals = sl.eigvals(Cmat, Sm)
            evals.sort()        # in-place!
        except:      # we give up, and place -1's in the return 1d-array
            evals = ones(n).A1 * (-1)
    # default axis in cumsum works here:
    return evals.cumsum() 


#############################################################
# test cases:
if __name__ == '__main__':
    from numpy.matlib import rand
    data = rand((100, 3))
    print getDeterministics(100, 'ctl', 0.3).shape
    print getDeterministics(100, 'c', 5).shape
    print getDeterministics(100, 'ctlsi', 80).shape
    print getDeterministics(100, 'qmtl', 0.1).shape
    print autocovar(data, 10)
    print autocovar(data, 5, True)
    print longrunvar(data)
    # the following could raise exceptions due to non-pos-def matrices
    print commontrendstest(data)
    print commontrendstest(data,2,'t',0.3)
    print commontrendstest(data,4,'2a',0.3)
    print commontrendstest(data,3,'1',0.3)
    try: print commontrendstest(data,5,'2',0.8)
    except: print '5 lags failed'
    # check loop for sorting evals:
    for run in range(100):
        m1 = rand((10,5))
        m2 = rand((10,5))
        m1in = m1.T * m1
        m2in = m2.T * m2
        evals = geneigsympos(m1in, m2in)[0]
        temp = evals
        temp.sort()
        for i in range(evals.shape[0]):
            if temp[i] != evals[i]:
                raise NotImplementedError, 'nono'
        #print run
    print geneigsympos(m1in, m2in)
    print getQuarterlyDates(1985, 3, 45)
    #print readGplFile('de-joint-wpu-6dim-1977q1.gpl')
    writecsv('test.csv', data, orientation = 'rows', \
     obslabels = ['one', 'two', 'three'], comments = ['hello', 'again'])
    print startobs2obslist('2000m12', 10)

'''
Changelog:
15May2007:
    add write_gretl_matrix_xml(),
    make commontrendstest() more robust to eigenvalue failures
1Feb2007:
    fix getOrthColumns to return a rx0 matrix if input full rank,
    and add a null() function for null spaces based on svd,
    rank() now directly based on svd
15Jan2007:
    new unvec() function
11Jan2007:
    new writecsv() function,
    deleted writeGpl...,
    new startobs2obslist(),
    new vec() function
10Jan2007:
    fixed use of c_ / r_ due to change in numpy API,
    fix bug in readcsv
7Jan2007:
    rewrote input checks using assert,
    generalized readcsv (formerly known as readgretlcsv)
5Jan2007:
    explicit sorting of eigenvalues instead of relying on numpy implementation
3Jan2007:
    new and simpler readgretlcsv (after gretl cvs changes),
    converter for numpy matrix to gretl-type matrix string
17Aug2006:
    fixes for readGplFile,
    finished writeGplFile
16Aug2006:
    removed obsolete qString2qNumber(),
    rewrote readGplFile with csv module, scipy or matplotlib not required,
    started analogous writeGplFile
15Aug2006:
    minor cosmetics
12Aug2006:
    added readGplFile,
    added getImpulseDummies
11Aug2006:
    added helpers for use with matplotlib and or gpl-formatted csv files,
    renamed getDetermMatrix to getDeterministics
10Aug2006:
    commented out diagm, can use .diagonal() and diagflat() in numpy
21Jul2006:
    commented out zerosm, onesm, emptym, eyem, randm, which are obsoleted 
     by the new numpy.matlib,
    what about diag?: still needs to be fixed in numpy,
    tried to avoid inefficient inverses (and use solve instead),
    replace asm/asmatrix by mat which now means the same in numpy,
    try to make makeNumpyMatrix redundant,
2Jun2006:
    added helpers zerosm, onesm, emptym, eyem, diagm to return numpy-matrices,
    added helper randm including workaround,
    switched to using ' instead of " where possible,
    don't add the replaced stuff like zeros etc. to the namespace anymore,
15May2006:
    kron is now in numpy, no need to substitute anymore;
    evals from geneigsympos are now always returned as array
1Mar2006:
    moved the Vecm class to file Vecmclass.py, and "stole" kron from scipy
28Feb2006:
    add Stock-Watson common-trends calculation
20Feb2006:
    work on deterministic adjustment of GG-decomp
14Feb2006:
    bugfixes and better treatment of S&L deterministics
12Feb2006:
    deterministics estimation a la S&L added to Vecm class,
    more use of numpy-lstsq-function 31Jan2006: all functions should
    return arrays or matrix-type according to the input, where that makes
    sense, i.e. whenever a data matrix is passed to a function(and where
    the purpose is not explicitly to produce matrices)
28Jan2006:
    bugfixing related to coeffs of restricted variables, wrote function
    for symmetric-def-gen.eigval problem to remove scipy-dependency
19Jan2006:
    work started on a vecm class 19Jan2006: switched over to
    raising exceptions instead of home-cooked string generation
19Jan2006:
    functions should all return numpy-matrix-type
6Jan2006:
    switched over to numpy/new-scipy
'''