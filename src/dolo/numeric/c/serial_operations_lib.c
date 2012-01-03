void cserop(int I, int J, int K, int N, double* A, double*B, double* C) {

    int rg, rm, rd;
    int i,j,k,n;
/*
    for(i=0;i<I;i++){
        for(k=0;k<K;k++) {
            for(j=0;j<J;j++) {
                for( n=0; n<N; n++) {
                    C[ i*N*K + k*N ] += A[rm]*B[rd];
                }
            }
        }
    }
*/

    for(i=0;i<I;i++){
        for(k=0;k<K;k++) {
            for(j=0;j<J;j++) {
                rg = i*N*K + k*N;
                rm = i*N*J+j*N;
                rd = j*N*K+k*N;
                for( n=0; n<N; n++) {
                    C[rg] += A[rm]*B[rd];
                    rg +=1;
                    rm +=1;
                    rd +=1;
                }
            }
        }
    }

}
