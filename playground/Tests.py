import unittest
import numpy as np

from helpers import LinAlgBasics as laBasics

class TestLinAlg(unittest.TestCase):
    def testPfaffian(self):
        print("Testing computation of Pfaffian determinants...")
        testMatrixA = np.array([[0,2,3,4,5,6],[-2,0,1,0,1,0],[-3,-1,0,2,2,2],[-4,0,-2,0,0,1],[-5,-1,-2,0,0,-4],[-6,0,-2,-1,4,0]])
        exactPfaffianA = -34
        computedPfaffianA = laBasics.getPfaffian(testMatrixA)
        self.assertAlmostEqual(exactPfaffianA, computedPfaffianA)

if __name__ == "__main__":
    unittest.main()