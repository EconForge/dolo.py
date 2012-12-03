#include <math.h>

void multilinear_interpolation_1d(int d, double* smin, double* smax, int* orders, double* V, int n_s, double* s, double* output) {

    int i;
    long q_0;

    double lam_0;


    double s_0, sn_0;
    double snt_0;

    int order_0 = orders[0];

#pragma omp parallel for
    for (i=0; i<n_s; i++) {

        s_0 = s[ i ];

        sn_0 = (s_0-smin[0])/(smax[0]-smin[0]);

        snt_0 = fmax( fmin( sn_0, 0.9999 ), 0.0 );

        q_0 = floor( snt_0 *(orders[0]-1) );

        lam_0 = sn_0*(order_0-1) - q_0;

        double v_0, v_1;


        v_0 = V[ q_0 ];
        v_1 = V[ q_0+1 ];


        output[i] = lam_0*v_1 + (1-lam_0)*v_0;

    };

};

void multilinear_interpolation_2d(int d, double* smin, double* smax, int* orders, double* V, int n_s, double* s, double* output) {

    int i;
    long q_0;
    long q_1;
    double lam_0;
    double lam_1;

    double s_0, s_1, sn_0, sn_1;
    double snt_0, snt_1;

    int order_0 = orders[0];
    int order_1 = orders[1];

#pragma omp parallel for
    for (i=0; i<n_s; i++) {
        
        s_0 = s[ i ];
        s_1 = s[ n_s + i ];


        sn_0 = (s_0-smin[0])/(smax[0]-smin[0]);
        sn_1 = (s_1-smin[1])/(smax[1]-smin[1]);

        snt_0 = fmax( fmin( sn_0, 0.9999 ), 0.0 );
        snt_1 = fmax( fmin( sn_1, 0.9999 ), 0.0 );

        q_0 = floor( snt_0 *(orders[0]-1) );
        q_1 = floor( snt_1 *(orders[1]-1) );
        
        lam_0 = sn_0*(order_0-1) - q_0; 
        lam_1 = sn_1*(order_1-1) - q_1;
        
        double v_00, v_01, v_10, v_11;

        v_00 = V[ order_1*(q_0) + q_1 ];
        v_01 = V[ order_1*(q_0) + q_1+1 ];
        v_10 = V[ order_1*(q_0+1) + q_1 ];
        v_11 = V[ order_1*(q_0+1) + q_1+1 ];

        output[i] = lam_0*(lam_1*v_11 + (1-lam_1)*v_10) + (1-lam_0)*(lam_1*v_01 + (1-lam_1)*v_00);

    };

};

void multilinear_interpolation_3d(int d, double* smin, double* smax, int* orders, double* V, int n_s, double* s, double* output) {

    int i;

    long q_0, q_1, q_2;

    double lam_0, lam_1, lam_2;

    double s_0, s_1, s_2;
    double sn_0, sn_1, sn_2;
    double snt_0, snt_1, snt_2;


    int order_0 = orders[0];
    int order_1 = orders[1];
    int order_2 = orders[2];

    //printf("orders : %i : %i : %i", order_0, order_1, order_2);
#pragma omp parallel for
    for (i=0; i<n_s; i++) {
        
        s_0 = s[ i ];
        s_1 = s[ n_s + i ];
        s_2 = s[ 2*n_s + i ];

        sn_0 = (s_0-smin[0])/(smax[0]-smin[0]);
        sn_1 = (s_1-smin[1])/(smax[1]-smin[1]);
        sn_2 = (s_2-smin[2])/(smax[2]-smin[2]);

        snt_0 = fmax( fmin( sn_0, 0.9999 ), 0.0 );
        snt_1 = fmax( fmin( sn_1, 0.9999 ), 0.0 );
        snt_2 = fmax( fmin( sn_2, 0.9999 ), 0.0 );

        q_0 = floor( snt_0 *(order_0-1) );
        q_1 = floor( snt_1 *(order_1-1) );
        q_2 = floor( snt_2 *(order_2-1) );
        
        lam_0 = sn_0*(order_0-1) - q_0; 
        lam_1 = sn_1*(order_1-1) - q_1;
        lam_2 = sn_2*(order_2-1) - q_2;

        //printf("lam_0 %f : lam_1 %f : lam_2 % f\n", lam_0, lam_1, lam_2);

        double v_000, v_001, v_010, v_011, v_100, v_101, v_110, v_111;

        v_000 = V[ order_2*order_1*(q_0) + order_2*q_1 + q_2 ];
        v_001 = V[ order_2*order_1*(q_0) + order_2*q_1 + q_2+1 ];

        v_010 = V[ order_2*order_1*(q_0) + order_2*(q_1+1) + q_2 ];
        v_011 = V[ order_2*order_1*(q_0) + order_2*(q_1+1) + q_2+1 ];
 
        v_100 = V[ order_2*order_1*(q_0+1) + order_2*q_1 + q_2 ];
        v_101 = V[ order_2*order_1*(q_0+1) + order_2*q_1 + q_2+1 ];

        v_110 = V[ order_2*order_1*(q_0+1) + order_2*(q_1+1) + q_2 ];
        v_111 = V[ order_2*order_1*(q_0+1) + order_2*(q_1+1) + q_2+1 ];

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
                     );

    };

};


void multilinear_interpolation_4d(int d, double* smin, double* smax, int* orders, double* V, int n_s, double* s, double* output) {

    int i;

    long q_0, q_1, q_2, q_3;

    double lam_0, lam_1, lam_2, lam_3;

    double s_0, s_1, s_2, s_3;
    double sn_0, sn_1, sn_2, sn_3;
    double snt_0, snt_1, snt_2, snt_3;


    int order_0 = orders[0];
    int order_1 = orders[1];
    int order_2 = orders[2];
    int order_3 = orders[3];

#pragma omp parallel for
    for (i=0; i<n_s; i++) {

        s_0 = s[ i ];
        s_1 = s[ n_s + i ];
        s_2 = s[ 2*n_s + i ];
        s_3 = s[ 3*n_s + i ];



        sn_0 = (s_0-smin[0])/(smax[0]-smin[0]);
        sn_1 = (s_1-smin[1])/(smax[1]-smin[1]);
        sn_2 = (s_2-smin[2])/(smax[2]-smin[2]);
        sn_3 = (s_3-smin[3])/(smax[3]-smin[3]);

        snt_0 = fmax( fmin( sn_0, 0.9999 ), 0.0 );
        snt_1 = fmax( fmin( sn_1, 0.9999 ), 0.0 );
        snt_2 = fmax( fmin( sn_2, 0.9999 ), 0.0 );
        snt_3 = fmax( fmin( sn_3, 0.9999 ), 0.0 );

        q_0 = floor( snt_0 *(order_0-1) );
        q_1 = floor( snt_1 *(order_1-1) );
        q_2 = floor( snt_2 *(order_2-1) );
        q_3 = floor( snt_3 *(order_3-1) );

        lam_0 = sn_0*(order_0-1) - q_0;
        lam_1 = sn_1*(order_1-1) - q_1;
        lam_2 = sn_2*(order_2-1) - q_2;
        lam_3 = sn_3*(order_3-1) - q_3;

        double v_0000, v_0001, v_0010, v_0011, v_0100, v_0101, v_0110, v_0111;
        double v_1000, v_1001, v_1010, v_1011, v_1100, v_1101, v_1110, v_1111;

        v_0000 = V[ order_3*order_1*order_2*(q_0) + order_2*order_3*(q_1) + order_3*(q_2) + q_3];
        v_0001 = V[ order_3*order_1*order_2*(q_0) + order_2*order_3*(q_1) + order_3*(q_2) + q_3+1];

        v_0010 = V[ order_3*order_1*order_2*(q_0) + order_2*order_3*(q_1) + order_3*(q_2+1) + q_3];
        v_0011 = V[ order_3*order_1*order_2*(q_0) + order_2*order_3*(q_1) + order_3*(q_2+1) + q_3+1];

        v_0100 = V[ order_3*order_1*order_2*(q_0) + order_2*order_3*(q_1+1) + order_3*(q_2) + q_3];
        v_0101 = V[ order_3*order_1*order_2*(q_0) + order_2*order_3*(q_1+1) + order_3*(q_2) + q_3+1];

        v_0110 = V[ order_3*order_1*order_2*(q_0) + order_2*order_3*(q_1+1) + order_3*(q_2+1) + q_3];
        v_0111 = V[ order_3*order_1*order_2*(q_0) + order_2*order_3*(q_1+1) + order_3*(q_2+1) + q_3+1];


        v_1000 = V[ order_3*order_1*order_2*(q_0+1) + order_2*order_3*(q_1) + order_3*q_2 + q_3];
        v_1001 = V[ order_3*order_1*order_2*(q_0+1) + order_2*order_3*(q_1) + order_3*q_2 + q_3+1];

        v_1010 = V[ order_3*order_1*order_2*(q_0+1) + order_2*order_3*(q_1) + order_3*(q_2+1) + q_3];
        v_1011 = V[ order_3*order_1*order_2*(q_0+1) + order_2*order_3*(q_1) + order_3*(q_2+1) + q_3+1];

        v_1100 = V[ order_3*order_1*order_2*(q_0+1) + order_2*order_3*(q_1+1) + order_3*q_2 + q_3];
        v_1101 = V[ order_3*order_1*order_2*(q_0+1) + order_2*order_3*(q_1+1) + order_3*q_2 + q_3+1];

        v_1110 = V[ order_3*order_1*order_2*(q_0+1) + order_2*order_3*(q_1+1) + order_3*(q_2+1) + q_3];
        v_1111 = V[ order_3*order_1*order_2*(q_0+1) + order_2*order_3*(q_1+1) + order_3*(q_2+1) + q_3+1];

        output[i] =
         lam_0 * (
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
         );

    };

};

