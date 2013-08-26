#include <math.h>
#include <stdio.h>

__global__ void multilinear_interpolation_1d(float *smin, float *smax, int *orders, float *V, int N, float *s, float *output) {
    
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= N) {
        return;
    };

    int q_0;
    float lam_0;


    float s_0, sn_0;
    float v_0, v_1;

    int order_0 = orders[0];


        s_0 = s[ i ];
        sn_0 = (s_0-smin[0])/(smax[0]-smin[0]);
        
        q_0 = (int) floor( sn_0 *(orders[0]-1) );
        if (q_0 < 0) {
            q_0 = 0;
        }
        else if (q_0 >= orders[0]-2) {
            q_0 = orders[0]-2;
        };

        lam_0 = sn_0*(order_0-1) - q_0;


        v_0 = V[ q_0 ];
        v_1 = V[ q_0+1 ];


        output[i] = lam_0*v_1 + (1-lam_0)*v_0;

    };
    



__global__ void multilinear_interpolation_2d(float *smin, float *smax, int *orders, float *V, int N, float *s, float *output) {

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    int q_0;
    int q_1;
    float lam_0;
    float lam_1;

    float s_0, s_1, sn_0, sn_1;
    float snt_0, snt_1;

    int order_0 = orders[0];
    int order_1 = orders[1];
        
        s_0 = s[ i ];
        s_1 = s[ N + i ];


        sn_0 = (s_0-smin[0])/(smax[0]-smin[0]);
        sn_1 = (s_1-smin[1])/(smax[1]-smin[1]);

        snt_0 = fmaxf( fminf( sn_0, 0.9999 ), 0.0 );
        snt_1 = fmaxf( fminf( sn_1, 0.9999 ), 0.0 );

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

__global__ void multilinear_interpolation_3d(float* smin, float* smax, int* orders, float* V, int N, float* s, float* output) {

    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    int q_0, q_1, q_2;

    float lam_0, lam_1, lam_2;

    float s_0, s_1, s_2;
    float sn_0, sn_1, sn_2;
    float snt_0, snt_1, snt_2;


    int order_0 = orders[0];
    int order_1 = orders[1];
    int order_2 = orders[2];

        
        s_0 = s[ i ];
        s_1 = s[ N + i ];
        s_2 = s[ 2*N + i ];

        sn_0 = (s_0-smin[0])/(smax[0]-smin[0]);
        sn_1 = (s_1-smin[1])/(smax[1]-smin[1]);
        sn_2 = (s_2-smin[2])/(smax[2]-smin[2]);

        snt_0 = fmaxf( fminf( sn_0, 0.9999 ), 0.0 );
        snt_1 = fmaxf( fminf( sn_1, 0.9999 ), 0.0 );
        snt_2 = fmaxf( fminf( sn_2, 0.9999 ), 0.0 );

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

