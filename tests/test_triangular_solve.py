import unittest

#from dolo.symbolic.symbolic import Variable,Parameter,TSymbol
#import pickle

class  TriangularSolveCase(unittest.TestCase):

    def test_solve_simple_system(self):

        from dolo.misc.triangular_solver import triangular_solver


        system = [
            [1,2,3],
            [2,3],
            [3],
            [],
            []
        ]

        solution = triangular_solver(system)

        assert( tuple( solution ) == (3,4,2,1,0) ) # it should have been left unmodified

    def test_solve_symbolic_system(self):
        from dolo.misc.triangular_solver import solve_triangular_system

        import sympy
        [w,x,y,z,t] = vars = sympy.symbols('w, x, y, z, t')

        eqs = [
            x + y + z + t,
            y + z,
            z,
            1,
            1
        ]

        sdict = {s:eqs[i] for i,s in enumerate(vars) }

        solution = solve_triangular_system(sdict)

        assert(solution[z]==1)
        assert(solution[t]==1)
        assert(solution[y]==1)
        assert(solution[x]==2)
        assert(solution[w]==5)



if __name__ == '__main__':

    unittest.main()