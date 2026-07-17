import numpy as np
import torch
from helpers import StatBasics

#If not stated stated otherwise, "electronLocations" always has dimensions [NUMBER_OF_BATCHES, NUMBER_OF_ELECTRONS, 3]

def getProjectionMatrices(electronLocations: torch.Tensor):
    #A projection matrix at point p with normal vector n is defined as P = I - nn^T.
    normals = torch.nn.functional.normalize(electronLocations, p=2, dim=2)
    return torch.eye(3).repeat(electronLocations.shape[0], electronLocations.shape[1], 1, 1) - torch.matmul(normals.unsqueeze(-1), normals.unsqueeze(-2))

def getElectronPairForces(electronLocations: torch.Tensor):
    numBatches = electronLocations.shape[0]
    numElectrons = electronLocations.shape[1]
    expandedElectronLocations = electronLocations.unsqueeze(2).expand(numBatches, numElectrons, numElectrons, 3).clone()
    distanceMatrix = expandedElectronLocations - torch.transpose(expandedElectronLocations, dim0=1, dim1=2)
    projectionMatrices = getProjectionMatrices(electronLocations).unsqueeze(2).expand(numBatches, numElectrons, numElectrons, 3, 3)
    projectedDistancesMatrix = torch.linalg.norm(torch.matmul(projectionMatrices, distanceMatrix), ord=2, dim=3)
    forceMatrix = 1. / projectedDistancesMatrix
    summedForces = torch.sum(forceMatrix.diagonal(offset=0, dim1=-1, dim2=-2).zero_(), dim=(1,2))   #TODO: Check
    return summedForces
#TODO: Add tests for this.

def getSphereCurvatureTerm(numBatches: int, numElectrons: int, sphereRadius: float):
    #For the sphere: Mean curvature is M = 1/r and Gauss curvature K = 1/r^2. So (M^2-K) = 0.
    #Keep this function only for future purposes when more general geometries will hopefully be available.
    return torch.zeros([numBatches, numElectrons])

def computeLocalEnergy(waveFunction: torch.nn.Module, electronLocations: torch.Tensor, sphereRadius: float, particleMass: float = 1.0):
    #The Hamiltonian has three terms: The surface laplace term, the electrostatic term and the term for the manifolds curvature which is constant 0 for a sphere.
    surfaceLaplacians = torch.div(getSurfaceLaplacian(waveFunction, electronLocations, sphereRadius), waveFunction.apply(electronLocations)) #TODO: Check division.
    electrostaicForces = getElectronPairForces(electronLocations)
    curvatureTerm = getSphereCurvatureTerm(electronLocations.shape[0], electronLocations.shape[1], sphereRadius)
    return -1. / (2. * particleMass) * torch.sum(surfaceLaplacians, dim=1) + 0.5 * torch.sum(electrostaicForces, dim=1) + 0.5 * torch.sum(curvatureTerm, dim=1)

def getExpectedValueFromWaveFunction():
    pass

def computeLocalEnergyGradient():
    pass

def estimateVariationalEnergy(waveFunction, numElectrons, sphereRadius):
    batchSize = 16
    electronSamples = torch.stack([StatBasics.sampleFromWaveFunction(waveFunction, numElectrons, sphereRadius) for _ in range(batchSize)])
    surfaceLaplacians = getSurfaceLaplacian(waveFunction, electronSamples, sphereRadius)

def getSurfaceGradient(waveFunction: torch.nn.Module, electronLocations: torch.Tensor):
    normals = torch.nn.functional.normalize(electronLocations, p=2, dim=2)
    waveNetworkParams = dict(waveFunction.named_parameters())

    def callableWaveFunction(locations):
        return torch.func.functional_call(waveFunction, waveNetworkParams, locations)

    callableGradient = torch.func.jacrev(callableWaveFunction)
    embeddedGradients = torch.func.vmap(callableGradient)(electronLocations)
    surfaceGradients = embeddedGradients - torch.matmul(normals.unsqueeze(-2), embeddedGradients.unsqueeze(-1)).squeeze(-1) * normals
    return surfaceGradients
    
def getSurfaceLaplacian(waveFunction: torch.nn.Module, electronLocations: torch.Tensor, sphereRadius: float):
    normals = torch.nn.functional.normalize(electronLocations, p=2, dim=2)
    waveNetworkParams = dict(waveFunction.named_parameters())

    def callableWaveFunction(locations):
        return torch.func.functional_call(waveFunction, waveNetworkParams, locations)
    
    def extractRelevantDerivatives(fullHessian: torch.Tensor):
        helperIndex = torch.arange(fullHessian.shape[0])
        return fullHessian[helperIndex,:,helperIndex,:]

    #Use "torch.func.jacrev(torch.func.jacrev(...))" instead of "torch.func.hessian" since "hessian" throws weird deprecation warning.
    callableHessian = torch.func.jacrev(torch.func.jacrev(callableWaveFunction))
    completeHessian = torch.func.vmap(callableHessian)(electronLocations)
    embeddedHessians = torch.func.vmap(extractRelevantDerivatives)(completeHessian)

    embeddedLaplacians = torch.sum(torch.diagonal(embeddedHessians, offset=0, dim1=-1, dim2=-2), -1)
    normalCorrection = torch.matmul(torch.matmul(normals.unsqueeze(-2), embeddedHessians), normals.unsqueeze(-1)).squeeze()
    callableGradient = torch.func.jacrev(callableWaveFunction)
    embeddedGradients = torch.func.vmap(callableGradient)(electronLocations)
    curvatureCorrection = 2. / sphereRadius * torch.matmul(normals.unsqueeze(-2), embeddedGradients.unsqueeze(-1)).squeeze()

    return embeddedLaplacians - normalCorrection - curvatureCorrection
    