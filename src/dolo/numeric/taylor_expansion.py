from numpy import atleast_2d, dot
from dolo.numeric.decision_rule import CallableDecisionRule


class TaylorExpansion(CallableDecisionRule):
    def __init__(self, *l):

        if len(l) == 1:
            # backward compatibility
            l = l[0]

        l = tuple(l)

        self.order = len(l) - 2
        self.coefs = l
        self.S_bar = l[0]
        self.X_bar = l[1]
        self.X_s = l[2]
        if self.order >= 2:
            self.X_ss = l[3]
        if self.order >= 3:
            self.X_sss = l[4]
        if self.order >= 4:
            raise Exception("Not implemented")

    def __getitem__(self, ind):

        l = [self.S_bar.copy()]
        l.append(self.X_bar[ind].copy())
        l.append(self.X_s[ind, ...].copy())
        if self.order >= 2:
            l.append(self.X_ss[ind, ...].copy())
        if self.order >= 3:
            l.append(self.X_sss[ind, ...].copy())
        return TaylorExpansion(l)

    def eval_s(self, points):

        if self.order == 1:
            return eval_te_order_1(self.S_bar, self.X_bar, self.X_s, points)
        elif self.order == 2:
            return eval_te_order_2(self.S_bar, self.X_bar, self.X_s, self.X_ss, points)
        elif self.order == 3:
            return eval_te_order_3(
                self.S_bar, self.X_bar, self.X_s, self.X_ss, self.X_sss, points
            )

    def eval_ms(self, m, s):

        return self.eval_s(s)


# helper functions

from numba import guvectorize


@guvectorize(["void(f8[:],f8[:],f8[:,:],f8[:],f8[:])"], "(s),(x),(x,s),(s)->(x)")
def eval_te_order_1(s0, x0, x1, points, out):

    S = points.shape[0]
    X = x0.shape[0]
    for n in range(X):
        out[n] = x0[n]
        for i in range(S):
            out[n] += x1[n, i] * (points[i] - s0[i])

    # return out


@guvectorize(
    ["void(f8[:],f8[:],f8[:,:],f8[:,:,:],f8[:],f8[:])"],
    "(s),(x),(x,s),(x,s,s),(s)->(x)",
)
def eval_te_order_2(s0, x0, x1, x2, points, out):

    S = points.shape[0]
    X = x0.shape[0]
    for n in range(X):
        out[n] = x0[n]
        for i in range(S):
            out[n] += x1[n, i] * (points[i] - s0[i])
            for j in range(S):
                out[n] += x2[n, i, j] * (points[i] - s0[i]) * (points[j] - s0[j]) / 2.0


@guvectorize(
    ["void(f8[:],f8[:],f8[:,:],f8[:,:,:], f8[:,:,:,:], f8[:],f8[:])"],
    "(s),(x),(x,s),(x,s,s),(x,s,s,s),(s)->(x)",
)
def eval_te_order_3(s0, x0, x1, x2, x3, points, out):

    S = points.shape[0]
    X = x0.shape[0]
    for n in range(X):
        out[n] = x0[n]
        for i in range(S):
            out[n] += x1[n, i] * (points[i] - s0[i])
            for j in range(S):
                out[n] += x2[n, i, j] * (points[i] - s0[i]) * (points[j] - s0[j]) / 2.0
                for k in range(S):
                    out[n] += (
                        x3[n, i, j, k]
                        * (points[i] - s0[i])
                        * (points[j] - s0[j])
                        * (points[k] - s0[k])
                        / 6.0
                    )


class CDR(TaylorExpansion):
    pass


def test_taylor_expansion():

    import numpy
    from numpy import array

    s0 = array([0.2, 0.4, 1.1])
    x0 = array([1.2, 0.9])

    N = 1000
    points = numpy.random.random((N, 3))

    X_s = numpy.random.random((2, 3))
    X_ss = numpy.random.random((2, 3, 3))
    X_sss = numpy.random.random((2, 3, 3, 3))
    dr1 = TaylorExpansion(s0, x0, X_s)  # , X_ss, X_sss])
    dr2 = TaylorExpansion(s0, x0, X_s, X_ss)  # , X_sss])
    dr3 = TaylorExpansion(s0, x0, X_s, X_ss, X_sss)

    out1 = dr1(points)
    out2 = dr2(points)
    out3 = dr3(points)

    out1_1d = dr1(points[0, :])
    out2_1d = dr2(points[0, :])
    out3_1d = dr3(points[0, :])

    assert abs(out1_1d - out1[0, :]).max() == 0
    assert abs(out2_1d - out2[0, :]).max() == 0
    assert abs(out3_1d - out3[0, :]).max() == 0

    ds = points - s0[None, :]
    ds2 = ds[:, :, None].repeat(3, axis=2)
    ds3 = ds2[:, :, :, None].repeat(3, axis=3)

    from numpy import dot

    verif1 = x0[None, :] + numpy.dot(ds, X_s.T)
    verif2 = dr2.__call2__(points)
    verif3 = dr3.__call2__(points)

    assert abs(out1 - verif1).max() < 1e-12
    assert abs(out2 - verif2).max() < 1e-12
    assert abs(out3 - verif3).max() < 1e-12


if __name__ == "__main__":

    test_taylor_expansion()
