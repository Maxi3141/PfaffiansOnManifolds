import unittest
import numpy as np
import torch

from computation import LinAlgBasics as laBasics
from computation import Physics as phys


class TestLinAlg(unittest.TestCase):
    def testPfaffian(self):
        testMatrixANumpy = np.array([[0,2,3,4,5,6],[-2,0,1,0,1,0],[-3,-1,0,2,2,2],[-4,0,-2,0,0,1],[-5,-1,-2,0,0,-4],[-6,0,-2,-1,4,0]])
        testMatrixATorch = torch.tensor([[0,2,3,4,5,6],[-2,0,1,0,1,0],[-3,-1,0,2,2,2],[-4,0,-2,0,0,1],[-5,-1,-2,0,0,-4],[-6,0,-2,-1,4,0]], dtype=torch.float)
        exactPfaffianA = -34
        computedPfaffianANumpy = laBasics.getPfaffianNumpy(testMatrixANumpy)
        self.assertAlmostEqual(exactPfaffianA, computedPfaffianANumpy)
        computedPfaffianATorch = laBasics.getPfaffian(testMatrixATorch)
        self.assertAlmostEqual(exactPfaffianA, computedPfaffianATorch.item(), places=5)

class SurfaceLaplacianTestNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(torch.sin(x[:,:,0]) + torch.exp(x[:,:,1]) * x[:,:,2]**2, dim=1, keepdim=True)

class SurfaceGradientTest(unittest.TestCase):
    def testSurfaceGradient(self):
        testWaveFunction = SurfaceLaplacianTestNeuralNetwork()
        unnormedEvaluationPoints = torch.tensor([[[1., 0., 0.], [1., 1., 0.]], [[1., 1., 1.], [0., -1., 0.]]])
        testedRadii = [1., 2.]
        exactSurfaceGradients = torch.tensor([[[[0., 0., 0.], [0.380122298537815, -0.380122298537815, 0.]], [[-0.324943065260291, -0.569084168251684, 0.894027233511974], [1., 0., 0.]]], [[[0., 0., 0.,], [0.0779718473826873, -0.0779718473826873, 0.]], [[-3.5834258344633, 0.243145785734844, 3.34028004872846], [1., 0., 0.]]]])
        for rIndex, r in enumerate(testedRadii):
            evaluationPoints = r * torch.nn.functional.normalize(unnormedEvaluationPoints, p=2, dim=2)
            computedSurfaceGradients = phys.getSurfaceGradient(testWaveFunction, evaluationPoints)
            discrepany = computedSurfaceGradients - exactSurfaceGradients[rIndex]
            self.assertAlmostEqual(torch.linalg.norm(discrepany).item(), 0.0, 5)

class SurfaceLaplacianTest(unittest.TestCase):
    def testSurfaceLaplacian(self):
        testWaveFunction = SurfaceLaplacianTestNeuralNetwork()
        unnormedEvaluationPoints = torch.tensor([[[1., 0., 0.], [1., 1., 0.]], [[1., 1., 1.], [0., -1., 0.]]])
        testedRadii = [1., 2.]
        exactSurfaceLaplacians = torch.tensor([[[0.919395388263720, 2.65626327384969], [-2.99244262500077, 0.735758882342885]], [[2.41614683654714, 7.62234894051760], [-5.35053222399308, 0.270670566473225]]])
        for rIndex, r in enumerate(testedRadii):
            evaluationPoints = r * torch.nn.functional.normalize(unnormedEvaluationPoints, p=2, dim=2)
            computedSurfaceLaplacians = phys.getSurfaceLaplacian(testWaveFunction, evaluationPoints, r)
            discrepany = computedSurfaceLaplacians - exactSurfaceLaplacians[rIndex]
            self.assertAlmostEqual(torch.linalg.norm(discrepany).item(), 0.0, 5)

#TODO: Adjust the log-version tests so that they use 2-electron-functions instead of the current 1-electron ones.
class LogSurfaceGradientTest(unittest.TestCase):
    def testLogSurfaceGradient(self):
        testWaveFunction = SurfaceLaplacianTestNeuralNetwork()
        unnormedEvaluationPoints = torch.tensor([[[1., 0., 0.]], [[1., 1., 0.]], [[1., 1., 1.]], [[0., -1., 0.]]])
        testedRadii = [1., 2.]
        exactSurfaceGradients = torch.tensor([[[[0., 0., 0.]], [[0.585130354848348, -0.585130354848348, 0.]], [[-0.285143742618018, -0.499382220912953, 0.784525963530971]], [[0., 0., 0.]]], [[[0., 0., 0.]], [[0.0789375731261391, -0.0789375731261392, 0]], [[-0.69642774782334, 0.0472546327939882, 0.649173115029352]], [[0., 0., 0.]]]])
        for rIndex, r in enumerate(testedRadii):
            evaluationPoints = r * torch.nn.functional.normalize(unnormedEvaluationPoints, p=2, dim=2)
            computedSurfaceGradients = phys.getSurfaceGradient(testWaveFunction, evaluationPoints, True)
            discrepany = computedSurfaceGradients - exactSurfaceGradients[rIndex]
            self.assertAlmostEqual(torch.linalg.norm(discrepany).item(), 0.0, 5)

class LogSurfaceLaplacianTest(unittest.TestCase):
    def testLogSurfaceLaplacian(self):
        testWaveFunction = SurfaceLaplacianTestNeuralNetwork()
        unnormedEvaluationPoints = torch.tensor([[[1., 0., 0.]], [[1., 1., 0.]], [[1., 1., 1.]], [[0., -1., 0.]]])
        testedRadii = [1., 2.]
        exactLogSurfaceLaplacians = torch.tensor([[1.09260496670312, 3.40408760019398, -3.57209607027618, 0.0], [2.65715786572742, 7.70429379301717, -1.94852976057954, 0.0]])
        for rIndex, r in enumerate(testedRadii):
            evaluationPoints = r * torch.nn.functional.normalize(unnormedEvaluationPoints, p=2, dim=2)
            computedLogSurfaceLaplacians = phys.getSurfaceLaplacian(testWaveFunction, evaluationPoints, r, True)
            discrepany = computedLogSurfaceLaplacians - exactLogSurfaceLaplacians[rIndex]
            self.assertAlmostEqual(torch.linalg.norm(discrepany).item(), 0.0, 5)

if __name__ == "__main__":
    unittest.main()