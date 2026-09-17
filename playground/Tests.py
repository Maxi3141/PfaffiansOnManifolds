import unittest
import numpy as np
import torch

#TODO: Remove below
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
#END REMOVE

from computation import LinAlgBasics as laBasics
from computation import Physics as phys
from computation import StatBasics

class TestLinAlg(unittest.TestCase):
    def testPfaffian(self):
        testMatrixANumpy = np.array([[0,2,3,4,5,6],[-2,0,1,0,1,0],[-3,-1,0,2,2,2],[-4,0,-2,0,0,1],[-5,-1,-2,0,0,-4],[-6,0,-2,-1,4,0]])
        testMatrixATorch = torch.tensor([[0,2,3,4,5,6],[-2,0,1,0,1,0],[-3,-1,0,2,2,2],[-4,0,-2,0,0,1],[-5,-1,-2,0,0,-4],[-6,0,-2,-1,4,0]], dtype=torch.float)
        exactPfaffianA = -34
        computedPfaffianANumpy = laBasics.getPfaffianNumpy(testMatrixANumpy)
        self.assertAlmostEqual(exactPfaffianA, computedPfaffianANumpy)
        computedPfaffianATorch = laBasics.getPfaffian(testMatrixATorch)
        self.assertAlmostEqual(exactPfaffianA, computedPfaffianATorch.item(), places=4)

        testMatrixBTemplate = torch.randn(2, 3, 4, 4, dtype=torch.float64)
        testMatrixBTorch = testMatrixBTemplate - torch.transpose(testMatrixBTemplate, dim0=-2, dim1=-1)
        exactDetB = torch.linalg.det(testMatrixBTorch)
        computedDetB = torch.pow(laBasics.getPfaffian(testMatrixBTorch), 2.)
        self.assertAlmostEqual(torch.norm(exactDetB - computedDetB).item(), 0.0)

class SurfaceLaplacianTestNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(torch.sin(x[:,:,0]) + torch.exp(x[:,:,1]) * x[:,:,2]**2, dim=1, keepdim=True)

class PositiveOnlyTestNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.pow(torch.sum(torch.sin(x[:,:,0]) + torch.exp(x[:,:,1]) * x[:,:,2]**2, dim=1, keepdim=True), 2.) + 1.0

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
            discrepancy = computedLogSurfaceLaplacians - exactLogSurfaceLaplacians[rIndex]
            self.assertAlmostEqual(torch.linalg.norm(discrepancy).item(), 0.0, 5)

class SurfaceLaplaceOverFunctionTest(unittest.TestCase):
    def testLaplaceOverFunction(self):
        testWaveFunction = PositiveOnlyTestNeuralNetwork()
        electronLocations = StatBasics.sampleUniformOnSphere(16, 5)

        classicComputation = phys.getSurfaceLaplacian(testWaveFunction, electronLocations, 1.0) / testWaveFunction(electronLocations)
        logComputation     = phys.getSurfaceLaplacianOverFunction(testWaveFunction, electronLocations, 1.0)
        discrepancy = torch.norm(classicComputation - logComputation, p=2.).item()
        self.assertAlmostEqual(discrepancy, 0.0, 5)

#Model probability density so that it is continuous and the probabiltity of landing on the top half of the sphere is 90%.

class SamplingTestNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = x[:,:,2]
        pSingleSqrt = torch.where(z > 0., torch.sqrt(7. / (10. * torch.pi) * z + 1. / (10. * torch.pi)), torch.sqrt(1. / (10. * torch.pi) * z + 1. / (10. * torch.pi)))
        pSqrt = torch.prod(pSingleSqrt, dim=-1)
        return pSqrt

#If everything is correct, the ratio of electrons on the top half should be 90% if the Metropolis-Hasting-Implementation works correctly.
#If the ratio is far below 90% (or above), either something is wrong or we just had bad luck (Many electrons on the bottom half are unlikely but possible). 
#TODO: Add some cutoff so that the test fails, if the cutoff is not reached.
class SamplingTest(unittest.TestCase):
    def testSampling(self):
        testWaveNetwork = SamplingTestNeuralNetwork()
        batchSize = 16
        numElectrons = 10
        sampledLocations = StatBasics.sampleFromWaveFunction(testWaveNetwork, batchSize, numElectrons, 1.0, 64)
        numTopHalfElectrons = torch.sum(torch.where(sampledLocations[:,:,2] > 0., 1., 0.))
        numTotalElectrons = batchSize * numElectrons
        ratioTopHalfElectrons = float(numTopHalfElectrons) / float(numTotalElectrons)
        print("SamplingTest: Roughly 0.9 of all electrons should be on the top half of the sphere. Measured: ratioTopHalfElectrons =", ratioTopHalfElectrons)

class ModeTestNeuralNetwork(torch.nn.Module):
    def __init__(self, modeTargets: torch.Tensor):
        super().__init__()
        if len(modeTargets.shape) != 2 or modeTargets.shape[1] != 3:
            raise Exception("\"modeTargets\" has invalid shape!")
        self.modeTargets = modeTargets

    def forward(self, x):
        return torch.sum(torch.matmul(x.unsqueeze(-2), self.modeTargets.unsqueeze(-1)).squeeze(dim=(-1,-2)), dim=-1) + float(x.shape[1])

class ModeComputationTest(unittest.TestCase):
    def testModeComputation(self):
        modeTargets = torch.tensor([[1., 0., 0.], [0., 1., 0.], [-1., -1, 0.], [1., -1., -1.]])
        modeTargets = torch.nn.functional.normalize(modeTargets, p=2., dim=-1)
        testNetwork = ModeTestNeuralNetwork(modeTargets)
        modeEstimate = StatBasics.computeModeOfWaveFunction(testNetwork, 1, 4, 1.)
        modeDiscrepancy = modeEstimate - modeTargets
        modeError = torch.linalg.norm(modeDiscrepancy)
        self.assertAlmostEqual(modeError.item(), 0.0, places=5)

class ElectroStaticForcesTest(unittest.TestCase):
    def testElectroStaticForces(self):
        electronLocations = torch.Tensor([[[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]])
        forcesComputed = phys.getElectronPairForces(electronLocations).squeeze()
        forcesExact = torch.tensor([1.0, 1.5, 1.0, 1.5])
        self.assertAlmostEqual(torch.linalg.norm(forcesComputed - forcesExact).item(), 0.0, 6)

if __name__ == "__main__":
    print("---Running tests for linear algebra, surface differential operators and sampling---")
    print("Warning: The test for sampling is non-deterministic and can fail by design when a statistical type I error for H0 = \"Implementation works\" occurs!")
    unittest.main()