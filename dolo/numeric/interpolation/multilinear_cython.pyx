from libc.math cimport fmin, fmax, floor

cimport numpy as np
import numpy as np

def multilinear_interpolation(np.ndarray[np.double_t, ndim=1] smin, np.ndarray[np.double_t, ndim=1] smax, np.ndarray[np.int_t, ndim=1] orders, np.ndarray[np.double_t, ndim=2] values, np.ndarray[np.double_t, ndim=2] s):

    cdef int d = np.size(s,0)
    cdef int n_s = np.size(s,1)
    cdef int n_v = np.size(values,0)

    cdef np.ndarray[np.double_t, ndim=2] result = np.zeros((n_v,n_s))
    cdef np.ndarray[np.double_t, ndim=1] vals
    cdef np.ndarray[np.double_t, ndim=1] res


    if d == 1:
        fun = multilinear_interpolation_1d
    elif d == 2:
        fun = multilinear_interpolation_2d
    elif d == 3:
        fun = multilinear_interpolation_3d
    elif d == 4:
        fun = multilinear_interpolation_4d

    for i in range(n_v):

        vals = values[i,:]
        res = result[i,:]

        fun(smin, smax, orders, vals, n_s, s, res)

    return result



cdef multilinear_interpolation_1d(np.ndarray[np.double_t, ndim=1] a_smin, np.ndarray[np.double_t, ndim=1] a_smax,
                                  np.ndarray[np.int_t, ndim=1] orders, np.ndarray[np.double_t, ndim=1] a_V,
                                  int n_s, np.ndarray[np.double_t, ndim=2] a_s, np.ndarray[np.double_t, ndim=1] a_output):

    cdef int d = 1

    cdef double* smin = <double*> a_smin.data
    cdef double* smax = <double*> a_smax.data
    cdef double* V = <double*> a_V.data
    cdef double* s = <double*> a_s.data
    cdef double* output = <double*> a_output.data



    cdef int i
    cdef int q_0

    cdef double lam_0


    cdef double s_0, sn_0
    cdef double snt_0

    cdef int order_0 = orders[0]
    cdef double v_0, v_1


    #pragma omp parallel for
    for i in range(n_s):

        s_0 = s[ i ]

        sn_0 = (s_0-smin[0])/(smax[0]-smin[0])

        snt_0 = fmax( fmin( sn_0, 0.9999 ), 0.0 )

        q_0 = <int> floor( snt_0 *(orders[0]-1) )

        lam_0 = sn_0*(order_0-1) - q_0



        v_0 = V[ q_0 ]
        v_1 = V[ q_0+1 ]


        output[i] = lam_0*v_1 + (1-lam_0)*v_0


cdef multilinear_interpolation_2d(np.ndarray[np.double_t, ndim=1] a_smin, np.ndarray[np.double_t, ndim=1] a_smax,
                                  np.ndarray[np.int_t, ndim=1] orders, np.ndarray[np.double_t, ndim=1] a_V,
                                  int n_s, np.ndarray[np.double_t, ndim=2] a_s, np.ndarray[np.double_t, ndim=1] a_output):

    cdef int d = 2

    cdef double* smin = <double*> a_smin.data
    cdef double* smax = <double*> a_smax.data
    cdef double* V = <double*> a_V.data
    cdef double* s = <double*> a_s.data
    cdef double* output = <double*> a_output.data

    cdef int i
    cdef int q_0
    cdef int q_1
    cdef double lam_0
    cdef double lam_1

    cdef double s_0, s_1, sn_0, sn_1
    cdef double snt_0, snt_1

    cdef int order_0 = orders[0]
    cdef int order_1 = orders[1]

    cdef double v_00, v_01, v_10, v_11

    #pragma omp parallel for
    for i in range(n_s):

        s_0 = s[ i ]
        s_1 = s[ n_s + i ]


        sn_0 = (s_0-smin[0])/(smax[0]-smin[0])
        sn_1 = (s_1-smin[1])/(smax[1]-smin[1])

        q_0 = max( min( <int>(sn_0 *(order_0-1)), (order_0-1) ), 0 )
        q_1 = max( min( <int>(sn_1 *(order_1-1)), (order_1-1) ), 0 )

        lam_0 = sn_0*(order_0-1) - q_0
        lam_1 = sn_1*(order_1-1) - q_1

        v_00 = V[ order_1*(q_0) + q_1 ]
        v_01 = V[ order_1*(q_0) + q_1+1 ]
        v_10 = V[ order_1*(q_0+1) + q_1 ]
        v_11 = V[ order_1*(q_0+1) + q_1+1 ]


        output[i] = lam_0*(lam_1*v_11 + (1-lam_1)*v_10) + (1-lam_0)*(lam_1*v_01 + (1-lam_1)*v_00)


cdef multilinear_interpolation_3d(np.ndarray[np.double_t, ndim=1] a_smin, np.ndarray[np.double_t, ndim=1] a_smax,
                                  np.ndarray[np.int_t, ndim=1] orders, np.ndarray[np.double_t, ndim=1] a_V,
                                  int n_s, np.ndarray[np.double_t, ndim=2] a_s, np.ndarray[np.double_t, ndim=1] a_output):

    cdef int d = 3

    cdef double* smin = <double*> a_smin.data
    cdef double* smax = <double*> a_smax.data
    cdef double* V = <double*> a_V.data
    cdef double* s = <double*> a_s.data
    cdef double* output = <double*> a_output.data


    cdef int i

    cdef int q_0, q_1, q_2

    cdef double lam_0, lam_1, lam_2

    cdef double s_0, s_1, s_2
    cdef double sn_0, sn_1, sn_2
    cdef double snt_0, snt_1, snt_2


    cdef int order_0 = orders[0]
    cdef int order_1 = orders[1]
    cdef int order_2 = orders[2]

    cdef double v_000, v_001, v_010, v_011, v_100, v_101, v_110, v_111


    #pragma omp parallel for
    for i in range(n_s):

        s_0 = s[ i ]
        s_1 = s[ n_s + i ]
        s_2 = s[ 2*n_s + i ]

        sn_0 = (s_0-smin[0])/(smax[0]-smin[0])
        sn_1 = (s_1-smin[1])/(smax[1]-smin[1])
        sn_2 = (s_2-smin[2])/(smax[2]-smin[2])

        snt_0 = fmax( fmin( sn_0, 0.9999 ), 0.0 )
        snt_1 = fmax( fmin( sn_1, 0.9999 ), 0.0 )
        snt_2 = fmax( fmin( sn_2, 0.9999 ), 0.0 )

        q_0 = <int> floor( snt_0 *(order_0-1) )
        q_1 = <int> floor( snt_1 *(order_1-1) )
        q_2 = <int> floor( snt_2 *(order_2-1) )

        lam_0 = sn_0*(order_0-1) - q_0
        lam_1 = sn_1*(order_1-1) - q_1
        lam_2 = sn_2*(order_2-1) - q_2


        v_000 = V[ order_2*order_1*(q_0) + order_2*q_1 + q_2 ]
        v_001 = V[ order_2*order_1*(q_0) + order_2*q_1 + q_2+1 ]

        v_010 = V[ order_2*order_1*(q_0) + order_2*(q_1+1) + q_2 ]
        v_011 = V[ order_2*order_1*(q_0) + order_2*(q_1+1) + q_2+1 ]

        v_100 = V[ order_2*order_1*(q_0+1) + order_2*q_1 + q_2 ]
        v_101 = V[ order_2*order_1*(q_0+1) + order_2*q_1 + q_2+1 ]

        v_110 = V[ order_2*order_1*(q_0+1) + order_2*(q_1+1) + q_2 ]
        v_111 = V[ order_2*order_1*(q_0+1) + order_2*(q_1+1) + q_2+1 ]

        output[i] =  lam_0 * (
                lam_1*(
                    lam_2*v_111 + (1-lam_2)*v_110)
                + (1-lam_1)*(
                    lam_2*v_101 + (1-lam_2)*v_100)
            )  + (1-lam_0)*(
                             lam_1*(
                                 lam_2*v_011 + (1-lam_2)*v_010
                             ) + (1-lam_1)*(
                                 lam_2*v_001 + (1-lam_2)*v_000
                             )
            )



cdef multilinear_interpolation_4d(np.ndarray[np.double_t, ndim=1] a_smin, np.ndarray[np.double_t, ndim=1] a_smax,
                                  np.ndarray[np.int_t, ndim=1] orders, np.ndarray[np.double_t, ndim=1] a_V,
                                  int n_s, np.ndarray[np.double_t, ndim=2] a_s, np.ndarray[np.double_t, ndim=1] a_output):

    cdef int d = 4

    cdef double* smin = <double*> a_smin.data
    cdef double* smax = <double*> a_smax.data
    cdef double* V = <double*> a_V.data
    cdef double* s = <double*> a_s.data
    cdef double* output = <double*> a_output.data


    cdef int i

    cdef int q_0, q_1, q_2, q_3

    cdef double lam_0, lam_1, lam_2, lam_3

    cdef double s_0, s_1, s_2, s_3
    cdef double sn_0, sn_1, sn_2, sn_3
    cdef double snt_0, snt_1, snt_2, snt_3


    cdef int order_0 = orders[0]
    cdef int order_1 = orders[1]
    cdef int order_2 = orders[2]
    cdef int order_3 = orders[3]

    cdef double v_0000, v_0001, v_0010, v_0011, v_0100, v_0101, v_0110, v_0111
    cdef double v_1000, v_1001, v_1010, v_1011, v_1100, v_1101, v_1110, v_1111

    #pragma omp parallel for
    for i in range(n_s):

        s_0 = s[ i ]
        s_1 = s[ n_s + i ]
        s_2 = s[ 2*n_s + i ]
        s_3 = s[ 3*n_s + i ]



        sn_0 = (s_0-smin[0])/(smax[0]-smin[0])
        sn_1 = (s_1-smin[1])/(smax[1]-smin[1])
        sn_2 = (s_2-smin[2])/(smax[2]-smin[2])
        sn_3 = (s_3-smin[3])/(smax[3]-smin[3])

        snt_0 = fmax( fmin( sn_0, 0.9999 ), 0.0 )
        snt_1 = fmax( fmin( sn_1, 0.9999 ), 0.0 )
        snt_2 = fmax( fmin( sn_2, 0.9999 ), 0.0 )
        snt_3 = fmax( fmin( sn_3, 0.9999 ), 0.0 )

        q_0 = <int> floor( snt_0 *(order_0-1) )
        q_1 = <int> floor( snt_1 *(order_1-1) )
        q_2 = <int> floor( snt_2 *(order_2-1) )
        q_3 = <int> floor( snt_3 *(order_3-1) )

        lam_0 = sn_0*(order_0-1) - q_0
        lam_1 = sn_1*(order_1-1) - q_1
        lam_2 = sn_2*(order_2-1) - q_2
        lam_3 = sn_3*(order_3-1) - q_3



        v_0000 = V[ order_3*order_1*order_2*(q_0) + order_2*order_3*(q_1) + order_3*(q_2) + q_3]
        v_0001 = V[ order_3*order_1*order_2*(q_0) + order_2*order_3*(q_1) + order_3*(q_2) + q_3+1]

        v_0010 = V[ order_3*order_1*order_2*(q_0) + order_2*order_3*(q_1) + order_3*(q_2+1) + q_3]
        v_0011 = V[ order_3*order_1*order_2*(q_0) + order_2*order_3*(q_1) + order_3*(q_2+1) + q_3+1]

        v_0100 = V[ order_3*order_1*order_2*(q_0) + order_2*order_3*(q_1+1) + order_3*(q_2) + q_3]
        v_0101 = V[ order_3*order_1*order_2*(q_0) + order_2*order_3*(q_1+1) + order_3*(q_2) + q_3+1]

        v_0110 = V[ order_3*order_1*order_2*(q_0) + order_2*order_3*(q_1+1) + order_3*(q_2+1) + q_3]
        v_0111 = V[ order_3*order_1*order_2*(q_0) + order_2*order_3*(q_1+1) + order_3*(q_2+1) + q_3+1]


        v_1000 = V[ order_3*order_1*order_2*(q_0+1) + order_2*order_3*(q_1) + order_3*q_2 + q_3]
        v_1001 = V[ order_3*order_1*order_2*(q_0+1) + order_2*order_3*(q_1) + order_3*q_2 + q_3+1]

        v_1010 = V[ order_3*order_1*order_2*(q_0+1) + order_2*order_3*(q_1) + order_3*(q_2+1) + q_3]
        v_1011 = V[ order_3*order_1*order_2*(q_0+1) + order_2*order_3*(q_1) + order_3*(q_2+1) + q_3+1]

        v_1100 = V[ order_3*order_1*order_2*(q_0+1) + order_2*order_3*(q_1+1) + order_3*q_2 + q_3]
        v_1101 = V[ order_3*order_1*order_2*(q_0+1) + order_2*order_3*(q_1+1) + order_3*q_2 + q_3+1]

        v_1110 = V[ order_3*order_1*order_2*(q_0+1) + order_2*order_3*(q_1+1) + order_3*(q_2+1) + q_3]
        v_1111 = V[ order_3*order_1*order_2*(q_0+1) + order_2*order_3*(q_1+1) + order_3*(q_2+1) + q_3+1]

        output[i] =  lam_0 * (
                lam_1 * (
                    lam_2*(
                        lam_3*v_1111 + (1-lam_3)*v_1110)
                    + (1-lam_2)*(
                        lam_3*v_1101 + (1-lam_3)*v_1100)
                )
                + (1-lam_1) * (
                    lam_2*(
                        lam_3*v_1011 + (1-lam_3)*v_1010)
                    + (1-lam_2)*(
                        lam_3*v_1001 + (1-lam_3)*v_1000)
                )
            ) + (1-lam_0) * (
                lam_1 * (
                    lam_2*(
                        lam_3*v_0111 + (1-lam_3)*v_0110)
                    + (1-lam_2)*(
                        lam_3*v_0101 + (1-lam_3)*v_0100)
                )
                + (1-lam_1) * (
                    lam_2*(
                        lam_3*v_0011 + (1-lam_3)*v_0010)
                    + (1-lam_2)*(
                        lam_3*v_0001 + (1-lam_3)*v_0000)
                )
            )

