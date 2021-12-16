import unittest

import numpy as np

import _icc


class TestCorrelation(unittest.TestCase):
    """Test correlation module"""


    def test_icc(self):
    
        
        matrix = [np.random.randint(-200,202, size=(i*20)) for i in range(1,6)]
        corr_icc = _icc.IntraClassCorrelationCoefficient(matrix)
        result = corr_icc.icc1()
        print(result)
        #self.assertEqual(resultdd


if __name__ == '__main__':
    unittest.main()
