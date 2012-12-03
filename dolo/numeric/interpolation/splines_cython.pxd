

cdef extern from "bspline.h":

    ctypedef struct Ugrid:
        double start
        double end
        int num

    ctypedef struct BCtype_d:
        int lCode
        int rCode

    ctypedef struct UBspline_1d_d:
        pass

    ctypedef struct UBspline_2d_d:
        pass

    ctypedef struct UBspline_3d_d:
        pass

    ctypedef struct multi_UBspline_1d_d:
        pass

    ctypedef struct multi_UBspline_2d_d:
        pass

    ctypedef struct multi_UBspline_3d_d:
        pass

    UBspline_1d_d * create_UBspline_1d_d (Ugrid x_grid, BCtype_d xBC, double *data)
    UBspline_2d_d * create_UBspline_2d_d (Ugrid x_grid, Ugrid y_grid, BCtype_d xBC, BCtype_d yBC, double *data)
    UBspline_3d_d * create_UBspline_3d_d (Ugrid x_grid, Ugrid y_grid, Ugrid z_grid, BCtype_d xBC, BCtype_d yBC, BCtype_d zBC,double *data)

    void eval_UBspline_1d_d (UBspline_1d_d * spline, double x, double* val) nogil
    void eval_UBspline_2d_d (UBspline_2d_d * spline, double x, double y, double* val) nogil
    void eval_UBspline_3d_d (UBspline_3d_d * spline, double x, double y, double z, double* val) nogil

    # multiple splines
    multi_UBspline_1d_d * create_multi_UBspline_1d_d (Ugrid x_grid, BCtype_d xBC, int num_splines)
    multi_UBspline_2d_d * create_multi_UBspline_2d_d (Ugrid x_grid, Ugrid y_grid, BCtype_d xBC, BCtype_d yBC, int num_splines)
    multi_UBspline_3d_d * create_multi_UBspline_3d_d (Ugrid x_grid, Ugrid y_grid, Ugrid z_grid, BCtype_d  xBC,  BCtype_d   yBC, BCtype_d   zBC, int num_splines)

    void set_multi_UBspline_1d_d (multi_UBspline_1d_d* spline, int spline_num, double *data)
    void set_multi_UBspline_2d_d (multi_UBspline_2d_d* spline, int spline_num, double *data)
    void set_multi_UBspline_3d_d (multi_UBspline_3d_d* spline, int spline_num, double *data)


    void eval_multi_UBspline_1d_d (multi_UBspline_1d_d *  spline, double x, double*  val) nogil
    void eval_multi_UBspline_1d_d_vg  (multi_UBspline_1d_d *  spline, double x, double*  val, double*  grad) nogil

    void eval_multi_UBspline_2d_d     (multi_UBspline_2d_d *  spline, double x, double y, double*  val) nogil
    void eval_multi_UBspline_2d_d_vg  (multi_UBspline_2d_d * spline, double x, double y, double*  val, double*  grad) nogil

    void eval_multi_UBspline_3d_d     (multi_UBspline_3d_d * spline, double x, double y, double z, double*  val) nogil
    void eval_multi_UBspline_3d_d_vg  (multi_UBspline_3d_d * spline, double x, double y, double z, double*  val, double*  grad) nogil

    void destroy_Bspline (void *spline)