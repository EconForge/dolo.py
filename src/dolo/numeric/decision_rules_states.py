from numpy import tile, dot
from dolo.numeric.tensor import mdot

class CDR:

    def __init__(self,l):
        self.order = len(l) - 2
        self.S_bar = l[0]
        self.X_bar = l[1]
        self.X_s = l[2]
        if self.order >= 2:
            self.X_ss = l[3]
        if self.order >= 3:
            self.X_sss = l[4]

    def __call__(self,points):
        n_s = points.shape[1]
        ds = points - tile( self.S_bar, (n_s,1) ).T
        choice =  dot(self.X_s, ds) +  tile( self.X_bar, (n_s,1) ).T
        n_ss = self.X_s.shape[1]
        if self.order == 2:
			for k in range(self.X_ss.shape[0]):
				for i in range(n_ss):
					for j in range(n_ss):
						choice[k,:] += self.X_ss[k,i,j]*ds[i,:]*ds[j,:]/2
        if self.order == 3:
            for i in range(n_s):
                choice[:,i] += mdot(self.X_ss,[ds[:,i],ds[:,i]]) / 2
                choice[:,i] += mdot(self.X_sss,[ds[:,i],ds[:,i],ds[:,i]]) / 6
        return choice

    def interpolate(self,x):
        return self.__call__(x)

