import unittest
import pickle

from trash.dolo.symbolic.symbolic import TSymbol

class  PicklingTestCase(unittest.TestCase):

    def test_pickle_variable(self):

        v = TSymbol('v')
        eq = v + v(1) + v(2)

        to_be_pickled = {
            'v': v(1),
            'eq': eq
        }
        
        save_string = pickle.dumps(to_be_pickled)

        loaded = pickle.loads(save_string)

        v =  loaded['v']
        print(v.__class__)
        print(v.date)
        print( (v + v)**2 )

        eqq =  loaded['eq']
        print(eqq )


if __name__ == '__main__':
    unittest.main()