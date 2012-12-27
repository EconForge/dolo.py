#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "bspline.h"


void evaluate_spline_1d(double* smin, double* smax, int* orders, double* data, int N, double* x, double* output) {

    double start = smin[0];
    double end = smax[0];
   
    Ugrid x_grid;
    x_grid.start = start;
    x_grid.end = end;
    x_grid.num = orders[0];
    
    BCtype_d bc;
    bc.lCode = NATURAL;
    bc.rCode = NATURAL;
    bc.lVal = 0.0;
    bc.rVal = 0.0;
    
    UBspline_1d_d* spline = create_UBspline_1d_d(x_grid, bc, data);

    for (int i=0; i<N; i++) {
        eval_UBspline_1d_d(spline, x[i], &output[i]);
    }
    destroy_Bspline(spline);

}

void evaluate_spline_2d(double* smin, double* smax, int* orders, double* data, int N, double* x, double* y, double* output) {

    Ugrid x_grid, y_grid;
    x_grid.start = smin[0];
    x_grid.end = smax[0];
    x_grid.num = orders[0];

    y_grid.start = smin[1];
    y_grid.end = smax[1];
    y_grid.num = orders[1];

   
    BCtype_d bc_x, bc_y;
    bc_x.lCode = NATURAL;
    bc_y.lCode = NATURAL;

    bc_x.rCode = NATURAL;
    bc_y.rCode = NATURAL;

//
    UBspline_2d_d* spline = create_UBspline_2d_d(x_grid, y_grid, bc_x, bc_y, data);

    #pragma omp parallel for
    for (int i=0; i<N; i++) {
        eval_UBspline_2d_d(spline, x[i], y[i], &output[i]);
    }

    destroy_Bspline(spline);
};

void evaluate_spline_3d(double* smin, double* smax, int* orders, double* data, int N, double* x, double* y, double* z, double* output) {

    Ugrid x_grid, y_grid, z_grid;
    x_grid.start = smin[0];
    x_grid.end = smax[0];
    x_grid.num = orders[0];

    y_grid.start = smin[1];
    y_grid.end = smax[1];
    y_grid.num = orders[1];

    z_grid.start = smin[2];
    z_grid.end = smax[2];
    z_grid.num = orders[2];
    
    BCtype_d bc_x, bc_y, bc_z;
    bc_x.lCode = NATURAL;
    bc_y.lCode = NATURAL;
    bc_z.lCode = NATURAL;

    bc_x.rCode = NATURAL;
    bc_y.rCode = NATURAL;
    bc_z.rCode = NATURAL;


    UBspline_3d_d* spline = create_UBspline_3d_d(x_grid, y_grid, z_grid, bc_x, bc_y, bc_z, data);

    #pragma omp parallel for
    for (int i=0; i<N; i++) {
        eval_UBspline_3d_d(spline, x[i], y[i], z[i],  &output[i]);
    }
    destroy_Bspline(spline);

};

void evaluate_spline_1d_g(double* smin, double* smax, int* orders, double* data, int N, double* x, double* output, double* grad) {

    double start = smin[0];
    double end = smax[0];

    Ugrid x_grid;
    x_grid.start = start;
    x_grid.end = end;
    x_grid.num = orders[0];

    BCtype_d bc;
    bc.lCode = NATURAL;
    bc.rCode = NATURAL;
    bc.lVal = 0.0;
    bc.rVal = 0.0;

    UBspline_1d_d* spline = create_UBspline_1d_d(x_grid, bc, data);

    for (int i=0; i<N; i++) {
        eval_UBspline_1d_d_vg(spline, x[i], &output[i], &grad[i]);
    }
    destroy_Bspline(spline);

}

void evaluate_spline_2d_g(double* smin, double* smax, int* orders, double* data, int N, double* x, double* y, double* output, double* grad) {

    Ugrid x_grid, y_grid;
    x_grid.start = smin[0];
    x_grid.end = smax[0];
    x_grid.num = orders[0];

    y_grid.start = smin[1];
    y_grid.end = smax[1];
    y_grid.num = orders[1];


    BCtype_d bc_x, bc_y;
    bc_x.lCode = NATURAL;
    bc_y.lCode = NATURAL;

    bc_x.rCode = NATURAL;
    bc_y.rCode = NATURAL;

//
    UBspline_2d_d* spline = create_UBspline_2d_d(x_grid, y_grid, bc_x, bc_y, data);

    #pragma omp parallel for
    for (int i=0; i<N; i++) {
        eval_UBspline_2d_d_vg(spline, x[i], y[i], &output[i], &grad[i*2]);
    }

    destroy_Bspline(spline);
};

void evaluate_spline_3d_g(double* smin, double* smax, int* orders, double* data, int N, double* x, double* y, double* z, double* output, double* grad) {

    Ugrid x_grid, y_grid, z_grid;
    x_grid.start = smin[0];
    x_grid.end = smax[0];
    x_grid.num = orders[0];

    y_grid.start = smin[1];
    y_grid.end = smax[1];
    y_grid.num = orders[1];

    z_grid.start = smin[2];
    z_grid.end = smax[2];
    z_grid.num = orders[2];

    BCtype_d bc_x, bc_y, bc_z;
    bc_x.lCode = NATURAL;
    bc_y.lCode = NATURAL;
    bc_z.lCode = NATURAL;

    bc_x.rCode = NATURAL;
    bc_y.rCode = NATURAL;
    bc_z.rCode = NATURAL;


    UBspline_3d_d* spline = create_UBspline_3d_d(x_grid, y_grid, z_grid, bc_x, bc_y, bc_z, data);

    #pragma omp parallel for
    for (int i=0; i<N; i++) {
        eval_UBspline_3d_d_vg(spline, x[i], y[i], z[i],  &output[i], &grad[i*3]);
    }
    destroy_Bspline(spline);

};

