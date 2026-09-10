import numpy as np
import torch
from computation import StatBasics
from nn import WaveFunction

#If not stated stated otherwise, "electronLocations" always has dimensions [BATCH_SIZE, NUMBER_OF_ELECTRONS, 3]

def getProjectionMatrices(electronLocations: torch.Tensor):
    #A projection matrix at point p with normal vector n is defined as P = I - nn^T.
    normals = torch.nn.functional.normalize(electronLocations, p=2, dim=2)
    return torch.eye(3).repeat(electronLocations.shape[0], electronLocations.shape[1], 1, 1) - torch.matmul(normals.unsqueeze(-1), normals.unsqueeze(-2))

def getElectronPairForces(electronLocations: torch.Tensor):
    batchSize = electronLocations.shape[0]
    numElectrons = electronLocations.shape[1]
    expandedElectronLocations = electronLocations.unsqueeze(2).expand(batchSize, numElectrons, numElectrons, 3).clone()
    distanceMatrix = (expandedElectronLocations - torch.transpose(expandedElectronLocations, dim0=1, dim1=2)).unsqueeze(-1)
    projectionMatrices = getProjectionMatrices(electronLocations).unsqueeze(2).expand(batchSize, numElectrons, numElectrons, 3, 3)
    projectedDistancesMatrix = torch.linalg.norm(torch.matmul(projectionMatrices, distanceMatrix), ord=2, dim=3)
    forceMatrix = (1. / projectedDistancesMatrix).squeeze(-1)
    helperIndex = torch.arange(numElectrons)
    forceMatrix[:, helperIndex, helperIndex] = 0.
    summedForces = torch.sum(forceMatrix, dim=-1)
    return summedForces
#TODO: Add tests for this.

def getSphereCurvatureTerm(batchSize: int, numElectrons: int, sphereRadius: float):
    #For the sphere: Mean curvature is M = 1/r and Gauss curvature is K = 1/r^2. So (M^2-K) = 0.
    #Keep this function only for future purposes when more general geometries will hopefully be available.
    return torch.zeros([batchSize, numElectrons])

def computeLocalEnergy(waveFunction: torch.nn.Module, electronLocations: torch.Tensor, sphereRadius: float, particleMass: float):
    #The Hamiltonian has three terms: The surface laplace term, the electrostatic term and the term for the manifolds curvature which is constant 0 for a sphere.

    kineticTerm = getSurfaceLaplacianOverFunction(waveFunction, electronLocations, sphereRadius)
    electrostaticForces = getElectronPairForces(electronLocations)
    curvatureTerm = getSphereCurvatureTerm(electronLocations.shape[0], electronLocations.shape[1], sphereRadius)

    return -1. / (2. * particleMass) * torch.sum(kineticTerm, dim=1) + 0.5 * torch.sum(electrostaticForces, dim=1) + 0.5 * torch.sum(curvatureTerm, dim=1)

def estimateExpectedLocalEnergy(waveFunction: WaveFunction.MultiElectronWaveFunction, numSamples: int = 16, presampledElectrons: torch.Tensor = None):
    if presampledElectrons == None:
        electronLocations = StatBasics.sampleFromWaveFunction(waveFunction, numSamples, waveFunction.numElectrons, waveFunction.sphereRadius)
    else:
        electronLocations = presampledElectrons
    return torch.sum(computeLocalEnergy(waveFunction, electronLocations, waveFunction.sphereRadius, waveFunction.particleMass)).item() / numSamples

def estimateVMCGradient(waveFunction: WaveFunction.MultiElectronWaveFunction, batchSize: int = 16, presampledElectrons: torch.Tensor = None):
    if presampledElectrons == None:
        expectedLocalEnergy = estimateExpectedLocalEnergy(waveFunction, batchSize)
        electronLocations = StatBasics.sampleFromWaveFunction(waveFunction, batchSize, waveFunction.numElectrons, waveFunction.sphereRadius)
    else:
        expectedLocalEnergy = estimateExpectedLocalEnergy(waveFunction, batchSize, presampledElectrons)
        electronLocations = presampledElectrons
    measuredLocalEnergy = computeLocalEnergy(waveFunction, electronLocations, waveFunction.sphereRadius, waveFunction.particleMass)
    localEnergyDiff = measuredLocalEnergy - expectedLocalEnergy
    networkLogGradients = waveFunction.getLogGradient(electronLocations)
    vmcGradient = [torch.zeros_like(subParameter) for subParameter in networkLogGradients[0]]
    for subParameterIndex in range(len(networkLogGradients[0])):
        for batchIndex in range(batchSize):
            vmcGradient[subParameterIndex] += localEnergyDiff[batchIndex] * networkLogGradients[batchIndex][subParameterIndex] / float(batchSize)
    return vmcGradient

def getSurfaceLaplacianOverFunction(waveFunction: torch.nn.Module, electronLocations: torch.Tensor, sphereRadius: float):
    normals = torch.nn.functional.normalize(electronLocations, p=2, dim=2)
    waveNetworkParams = dict(waveFunction.named_parameters())
    logStabilizer = 1e-8

    def callableLogWaveFunction(locations):
            return torch.log(torch.abs(torch.func.functional_call(waveFunction, waveNetworkParams, locations.unsqueeze(0))) + logStabilizer)

    callableLogGradient = torch.func.jacfwd(callableLogWaveFunction)
    embeddedLogGradients = torch.func.vmap(callableLogGradient)(electronLocations).squeeze(dim=(1,2))

    def extractRelevantDerivatives(fullHessian: torch.Tensor):
        helperIndex = torch.arange(fullHessian.shape[0])
        return fullHessian[helperIndex,:,helperIndex,:]
    
    callableLogHessian = torch.func.jacfwd(torch.func.jacfwd(callableLogWaveFunction))
    completeLogHessian = torch.func.vmap(callableLogHessian)(electronLocations).squeeze(dim=(1,2))
    embeddedLogHessians = torch.func.vmap(extractRelevantDerivatives)(completeLogHessian)

    hessianDivFunction = embeddedLogHessians + torch.matmul(embeddedLogGradients.unsqueeze(-1), embeddedLogGradients.unsqueeze(-2))

    laplaceTerm = torch.sum(torch.diagonal(embeddedLogHessians, offset=0, dim1=-1, dim2=-2), -1).squeeze(-1) + torch.pow(torch.norm(embeddedLogGradients, p=2., dim=-1), 2.)
    hessianTerm = torch.matmul(torch.matmul(normals.unsqueeze(-2), hessianDivFunction), normals.unsqueeze(-1)).squeeze(dim=(-1, -2))
    gradientTerm = 2. / sphereRadius * torch.matmul(normals.unsqueeze(-2), embeddedLogGradients.unsqueeze(-1)).squeeze()

    return laplaceTerm - hessianTerm - gradientTerm

def getSurfaceGradient(waveFunction: torch.nn.Module, electronLocations: torch.Tensor, useLogScaling: bool = False):
    normals = torch.nn.functional.normalize(electronLocations, p=2, dim=2)
    waveNetworkParams = dict(waveFunction.named_parameters())
    logStabilizer = 1e-8

    def callableWaveFunction(locations):
        if useLogScaling:
            return torch.log(torch.abs(torch.func.functional_call(waveFunction, waveNetworkParams, locations.unsqueeze(0))) + logStabilizer)
        else:
            return torch.func.functional_call(waveFunction, waveNetworkParams, locations.unsqueeze(0))

    callableGradient = torch.func.jacfwd(callableWaveFunction)
    embeddedGradients = torch.func.vmap(callableGradient)(electronLocations).squeeze(dim=(1,2))
    surfaceGradients = embeddedGradients - torch.matmul(normals.unsqueeze(-2), embeddedGradients.unsqueeze(-1)).squeeze(-1) * normals
    return surfaceGradients
    
def getSurfaceLaplacian(waveFunction: torch.nn.Module, electronLocations: torch.Tensor, sphereRadius: float, useLogScaling: bool = False):
    normals = torch.nn.functional.normalize(electronLocations, p=2, dim=-1)
    waveNetworkParams = dict(waveFunction.named_parameters())
    logStabilizer = 1e-8

    def callableWaveFunction(locations):
        if useLogScaling:
            return torch.log(torch.abs(torch.func.functional_call(waveFunction, waveNetworkParams, locations.unsqueeze(0))) + logStabilizer)
        else:
            return torch.func.functional_call(waveFunction, waveNetworkParams, locations.unsqueeze(0))
    
    def extractRelevantDerivatives(fullHessian: torch.Tensor):
        helperIndex = torch.arange(fullHessian.shape[0])
        return fullHessian[helperIndex,:,helperIndex,:]

    #Using "torch.func.jacrev(torch.func.jacrev(...))" instead of "torch.func.hessian(...)" since the latter throws a weird deprecation warning.
    callableHessian = torch.func.jacfwd(torch.func.jacfwd(callableWaveFunction))
    completeHessian = torch.func.vmap(callableHessian)(electronLocations).squeeze(dim=(1,2))
    embeddedHessians = torch.func.vmap(extractRelevantDerivatives)(completeHessian)

    embeddedLaplacians = torch.sum(torch.diagonal(embeddedHessians, offset=0, dim1=-1, dim2=-2), -1).squeeze(-1)
    normalCorrection = torch.matmul(torch.matmul(normals.unsqueeze(-2), embeddedHessians), normals.unsqueeze(-1)).squeeze()
    callableGradient = torch.func.jacfwd(callableWaveFunction)
    embeddedGradients = torch.func.vmap(callableGradient)(electronLocations).squeeze(dim=(1,2))
    curvatureCorrection = 2. / sphereRadius * torch.matmul(normals.unsqueeze(-2), embeddedGradients.unsqueeze(-1)).squeeze()

    result = embeddedLaplacians - normalCorrection - curvatureCorrection
    return result

#This function computes the energy in terms of the Thomson problem of the electrons for every batch and then returns
#the average energy over all batches, the lowest energy of any batch and the highest energy of any batch in that order in a 3-list.
def getAvgLowHighThomsonEnergy(electronLocations: torch.Tensor):
    batchSize = electronLocations.shape[0]
    numElectrons = electronLocations.shape[1]
    expandedElectronLocations = electronLocations.unsqueeze(2).expand(batchSize, numElectrons, numElectrons, 3).clone()
    distanceMatrix = torch.linalg.norm(expandedElectronLocations - torch.transpose(expandedElectronLocations, dim0=1, dim1=2), ord=2, dim=3)
    #print("distanceMatrix =", distanceMatrix)
    energyMatrix = (1. / distanceMatrix)
    helperIndex = torch.arange(numElectrons)
    energyMatrix[:, helperIndex, helperIndex] = 0.
    batchEnergies = torch.sum(energyMatrix, dim=(1,2)) / 2.
    highEnergy = torch.max(batchEnergies).item()
    lowEnergy = torch.min(batchEnergies).item()
    avgEnergy = torch.sum(batchEnergies).item() / float(batchSize)
    return [avgEnergy, lowEnergy, highEnergy]

def getAvgLowHighLocalEnergy(waveFunction: WaveFunction.MultiElectronWaveFunction, electronLocations: torch.Tensor, sphereRadius: float, particleMass: float = 1.0):
    batchSize = electronLocations.shape[0]
    batchEnergies = computeLocalEnergy(waveFunction, electronLocations, waveFunction.sphereRadius, particleMass)
    highEnergy = torch.max(batchEnergies).item()
    lowEnergy = torch.min(batchEnergies).item()
    avgEnergy = torch.sum(batchEnergies).item() / float(batchSize)
    return [avgEnergy, lowEnergy, highEnergy]