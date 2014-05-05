from numpy import atleast_2d, dot

class TaylorExpansion:

    def __init__(self,l):
        self.order = len(l) - 2
        self.S_bar = l[0]
        self.X_bar = l[1]
        self.X_s = l[2]
        if self.order >= 2:
            raise Exception("Not implemented")
            self.X_ss = l[3]
        if self.order >= 3:
            self.X_sss = l[4]

    def __getitem__(self, ind):

        l = [ self.S_bar.copy() ]
        l.append(self.X_bar[ind].copy())
        l.append(self.X_s[ind,...].copy())
        if self.order >= 2:
            l.append(self.X_ss[ind,...].copy())
        if self.order >= 3:
            l.append(self.X_sss[ind,...].copy())
        return CDR( l )


    def __call__(self, points):

        if points.ndim == 1:
            pp = atleast_2d(points)
            res = self.__call__(pp)
            return res.ravel()

        n_s = points.shape[0]

        ds = points - self.S_bar[None,:]
        choice = self.X_bar[None,:] + dot( ds, self.X_s.T )

        return choice


class CDR(TaylorExpansion):
    # for compatibility
    pass