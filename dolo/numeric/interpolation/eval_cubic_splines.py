from numba import jit

# dispatch on dimensions

def eval_cubic_spline(a, b, orders, coefs, point, value):

	pass

def eval_cubic_spline_d(a, b, orders, coefs, point, value, dvalue):

	pass




def eval_cubic_multi_spline(a, b, orders, mcoefs, point, values):

	pass

def eval_cubic_multi_spline_d(a, b, orders, mcoefs, point, values, dvalues):

	pass





@jit
def vec_eval_cubic_multi_spline(a, b, orders, mcoefs, points, values):

	from eval_splines_numba import eval_cubic_multi_spline_1, eval_cubic_multi_spline_2, eval_cubic_multi_spline_3
	from eval_splines_numba import Ad, dAd

	d = a.shape[0]
	N = points.shape[0]

	if d == 1:

		for n in range(N):
			point = points[n,:]
			val = values [n,:]
			eval_cubic_multi_spline_1(a, b, orders, mcoefs, point, val, Ad, dAd)

	elif d == 2:

		for n in range(N):
			point = points[n,:]
			val = values [n,:]
			eval_cubic_multi_spline_2(a, b, orders, mcoefs, point, val, Ad, dAd)

	elif d == 3:

		for n in range(N):
			point = points[n,:]
			val = values [n,:]
			eval_cubic_multi_spline_3(a, b, orders, mcoefs, point, val, Ad, dAd)




def vec_eval_cubic_multi_spline_d(a, b, orders, mcoefs, points, values, dvalues):

	pass


