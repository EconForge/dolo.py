from __future__ import division

import unittest


class  TestInterpolation(unittest.TestCase):

    def test_smolyak(self):

        import numpy

        f = lambda x: numpy.row_stack([
            x[0,:] * x[1,:]**0.5,
            x[1,:] * x[1,:] - x[0,:] * x[0,:]
        ])

        bounds = numpy.row_stack([
            [0.5,0.1],
            [2,3]
        ])

        from dolo.numeric.smolyak import SmolyakGrid
        sg = SmolyakGrid(bounds,3)

        values = f(sg.grid)
        sg.fit_values(values)
        theta_0 = sg.theta.copy()

        def fobj(theta):
            sg.theta = theta
            return sg(sg.grid)

        fobj(theta_0)

    def test_smolyak_2(self):

        import numpy
        from dolo.numeric.smolyak import SmolyakGrid
        d = 8
        l = 4

        bounds = numpy.row_stack([[-0.5]*6, [0.7]*6])
        sg = SmolyakGrid(bounds,l)
        f = lambda x: numpy.row_stack([
                    x[0,:] * x[1,:],
                    x[1,:] * x[1,:] - x[0,:] * x[0,:]
                ])
        values = f(sg.grid)

        import time
        t = time.time()
        for i in range(5):
            sg.fit_values(sg.grid)

            val = sg(sg.grid)
        s = time.time()
        print(s-t)

        
if __name__ == '__main__':

    import numpy
    from dolo.numeric.smolyak import SmolyakGrid
    d = 8
    l = 4

    bounds = numpy.row_stack([[-0.5]*6, [0.7]*6])
    sg = SmolyakGrid(bounds,l)
    f = lambda x: numpy.row_stack([
                x[0,:] * x[1,:],
                x[1,:] * x[1,:] - x[0,:] * x[0,:]
            ])
    values = f(sg.grid)
    tt = numpy.repeat(sg.grid,40,axis=1)

    print tt.shape
    import time
    t = time.time()
    sg.fit_values(sg.grid)

    for i in range(5):

        val = sg(tt)
    s = time.time()
    print(s-t)
#    unittest.main()
#    tt = TestInterpolation()
#    #    tt.test_2d_interpolation()
#    tt.test_smolyak_2()

