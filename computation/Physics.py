import numpy as np
import torch
from computation import StatBasics
from nn import WaveFunction
from computation import SphereManifold
from collections.abc import Callable

#If not stated stated otherwise, "electronLocations" always has dimensions [BATCH_SIZE, NUMBER_OF_ELECTRONS, 3]

#TODO: Update all the old calls to WaveFunction.getLogGradient to the new format where it returns list[torch.Tensor] with each tensor being batched.
#TODO: Simplify all the if-clauses that differentiate between "sphereTransform == None" and "sphereTransform != Noen"

def getProjectionMatrices(electronLocations: torch.Tensor):
    #A projection matrix at point p with normal vector n is defined as P = I - nn^T.
    normals = torch.nn.functional.normalize(electronLocations, p=2, dim=2)
    return torch.eye(3).repeat(electronLocations.shape[0], electronLocations.shape[1], 1, 1) - torch.matmul(normals.unsqueeze(-1), normals.unsqueeze(-2))

def getElectronPairForces(electronSphereLocations: torch.Tensor, sphereTransform: Callable[[torch.Tensor], torch.Tensor]):
    batchSize = electronSphereLocations.shape[0]
    numElectrons = electronSphereLocations.shape[1]

    if sphereTransform == None:
        expandedElectronLocations = electronSphereLocations.unsqueeze(2).expand(batchSize, numElectrons, numElectrons, 3).clone()
    else:
        expandedElectronLocations = sphereTransform(electronSphereLocations).unsqueeze(2).expand(batchSize, numElectrons, numElectrons, 3).clone()

    distanceMatrix = (expandedElectronLocations - torch.transpose(expandedElectronLocations, dim0=1, dim1=2)).unsqueeze(-1)

    if sphereTransform == None:
        projectionMatrices = getProjectionMatrices(electronSphereLocations).unsqueeze(2).expand(batchSize, numElectrons, numElectrons, 3, 3)
    else:
        projectionMatrices = SphereManifold.getManifoldProjectionsFromSphereCoords(electronSphereLocations, sphereTransform).unsqueeze(2).expand(batchSize, numElectrons, numElectrons, 3, 3)

    projectedDistancesMatrix = torch.linalg.norm(torch.matmul(projectionMatrices, distanceMatrix), ord=2, dim=3)
    embeddedDistancesMatrix = torch.linalg.norm(distanceMatrix, ord=2, dim=3)
    forceMatrix = (projectedDistancesMatrix / torch.pow(embeddedDistancesMatrix, 2.)).squeeze(-1)
    helperIndex = torch.arange(numElectrons)
    forceMatrix[:, helperIndex, helperIndex] = 0.
    summedForces = torch.sum(forceMatrix, dim=-1)
    return summedForces

def getSphereCurvatureTerm(electronSphereLocations: torch.Tensor, sphereTransform: Callable[[torch.Tensor], torch.Tensor]):
    #For the sphere: Mean curvature is M = 1/r and Gauss curvature is K = 1/r^2. So (M^2-K) = 0.
    if sphereTransform == None:
        return torch.zeros([electronSphereLocations.shape[0], electronSphereLocations.shape[1]])
    else:
        return SphereManifold.getGeometricPotential(electronSphereLocations, sphereTransform)

def computeLocalEnergy(waveFunction: torch.nn.Module, electronSphereLocations: torch.Tensor, sphereTransform: Callable[[torch.Tensor], torch.Tensor], particleMass: float):
    #The Hamiltonian has three terms: The surface laplace term, the electrostatic term and the term for the manifolds curvature which is constant 0 for a sphere.

    kineticTerm = getSurfaceLaplacianOverFunction(waveFunction, electronSphereLocations, sphereTransform)
    electrostaticForces = getElectronPairForces(electronSphereLocations, sphereTransform)
    curvatureTerm = getSphereCurvatureTerm(electronSphereLocations, sphereTransform)

    return -1. / (2. * particleMass) * torch.sum(kineticTerm, dim=1) + 0.5 * torch.sum(electrostaticForces, dim=1) - 0.5 * torch.sum(curvatureTerm, dim=1)

def estimateExpectedLocalEnergy(waveFunction: WaveFunction.MultiElectronWaveFunction, numSamples: int = 16, presampledElectrons: torch.Tensor = None):
    if presampledElectrons == None:
        electronLocations = StatBasics.sampleFromWaveFunction(waveFunction, numSamples, waveFunction.numElectrons, waveFunction.sphereRadius)
    else:
        electronLocations = presampledElectrons
    return torch.sum(computeLocalEnergy(waveFunction, electronLocations, waveFunction.sphereRadius, waveFunction.particleMass)).item() / numSamples

def estimateVMCGradient(waveFunction: WaveFunction.MultiElectronWaveFunction, batchSize: int = 16, presampledElectrons: torch.Tensor = None):
    raise Exception("\"estimateVMCGradient\" should not have been called. Use Spring-Optimizer from LinAlgBasics directly instead.")
    if presampledElectrons == None:
        #expectedLocalEnergy = estimateExpectedLocalEnergy(waveFunction, batchSize)
        electronLocations = StatBasics.sampleFromWaveFunction(waveFunction, batchSize, waveFunction.numElectrons, waveFunction.sphereRadius)
    else:
        #expectedLocalEnergy = estimateExpectedLocalEnergy(waveFunction, batchSize, presampledElectrons)
        electronLocations = presampledElectrons
    measuredLocalEnergy = computeLocalEnergy(waveFunction, electronLocations, waveFunction.sphereRadius, waveFunction.particleMass)
    expectedLocalEnergy = torch.sum(measuredLocalEnergy) / batchSize
    
    localEnergyDiff = measuredLocalEnergy - expectedLocalEnergy
    networkLogGradients = waveFunction.getLogGradient(electronLocations)
    vmcGradient = [torch.zeros(subParameter) for subParameter in networkLogGradients[0]]
    for subParameterIndex in range(len(networkLogGradients[0])):
        for batchIndex in range(batchSize):
            vmcGradient[subParameterIndex] += localEnergyDiff[batchIndex] * networkLogGradients[batchIndex][subParameterIndex] / float(batchSize)
    return vmcGradient

def getSurfaceLaplacianOverFunction(waveFunction: torch.nn.Module, electronSphereLocations: torch.Tensor, sphereTransform: Callable[[torch.Tensor], torch.Tensor]):

    if sphereTransform == None:
        normals = torch.nn.functional.normalize(electronSphereLocations, p=2, dim=2)
        electronManifoldLocations = electronSphereLocations
    else: 
        normals = SphereManifold.getManifoldNormalsFromSphereCoords(electronSphereLocations, sphereTransform)
        electronManifoldLocations = sphereTransform(electronSphereLocations)

    waveNetworkParams = dict(waveFunction.named_parameters())
    logStabilizer = 1e-8

    def callableLogWaveFunction(locations):
        return torch.log(torch.abs(torch.func.functional_call(waveFunction, waveNetworkParams, locations.unsqueeze(0))) + logStabilizer)

    callableLogGradient = torch.func.jacfwd(callableLogWaveFunction)
    embeddedLogGradients = torch.func.vmap(callableLogGradient, randomness='different')(electronManifoldLocations).squeeze(dim=(1,2))

    def extractRelevantDerivatives(fullHessian: torch.Tensor):
        helperIndex = torch.arange(fullHessian.shape[0])
        return fullHessian[helperIndex,:,helperIndex,:]
    
    callableLogHessian = torch.func.jacfwd(torch.func.jacfwd(callableLogWaveFunction))
    completeLogHessian = torch.func.vmap(callableLogHessian, randomness="different")(electronManifoldLocations).squeeze(dim=(1,2))
    embeddedLogHessians = torch.func.vmap(extractRelevantDerivatives, randomness="different")(completeLogHessian)

    hessianDivFunction = embeddedLogHessians + torch.matmul(embeddedLogGradients.unsqueeze(-1), embeddedLogGradients.unsqueeze(-2))

    if sphereTransform == None:
        totalCurvature = 2.
    else:
        totalCurvature = SphereManifold.getTotalCurvature(electronSphereLocations, sphereTransform)

    laplaceTerm = torch.sum(torch.diagonal(embeddedLogHessians, offset=0, dim1=-1, dim2=-2), -1).squeeze(-1) + torch.pow(torch.norm(embeddedLogGradients, p=2., dim=-1), 2.)
    hessianTerm = torch.matmul(torch.matmul(normals.unsqueeze(-2), hessianDivFunction), normals.unsqueeze(-1)).squeeze(dim=(-1, -2))
    gradientTerm = totalCurvature * torch.matmul(normals.unsqueeze(-2), embeddedLogGradients.unsqueeze(-1)).squeeze()

    return laplaceTerm - hessianTerm - gradientTerm

def getSurfaceGradient(waveFunction: torch.nn.Module, electronSphereLocations: torch.Tensor, sphereTransform: Callable[[torch.Tensor], torch.Tensor], useLogScaling: bool = False):
    electronManifoldLocations = sphereTransform(electronSphereLocations)

    if sphereTransform == None:
        normals = torch.nn.functional.normalize(electronSphereLocations, p=2, dim=2)
    else:
        normals = SphereManifold.getManifoldNormalsFromSphereCoords(electronSphereLocations)

    waveNetworkParams = dict(waveFunction.named_parameters())
    logStabilizer = 1e-8

    def callableWaveFunction(locations):
        if useLogScaling:
            return torch.log(torch.abs(torch.func.functional_call(waveFunction, waveNetworkParams, locations.unsqueeze(0))) + logStabilizer)
        else:
            return torch.func.functional_call(waveFunction, waveNetworkParams, locations.unsqueeze(0))

    callableGradient = torch.func.jacfwd(callableWaveFunction)
    embeddedGradients = torch.func.vmap(callableGradient, randomness="different")(electronManifoldLocations).squeeze(dim=(1,2))
    surfaceGradients = embeddedGradients - torch.matmul(normals.unsqueeze(-2), embeddedGradients.unsqueeze(-1)).squeeze(-1) * normals
    return surfaceGradients
    
def getSurfaceLaplacian(waveFunction: torch.nn.Module, electronSphereLocations: torch.Tensor, sphereTransform: Callable[[torch.Tensor], torch.Tensor], useLogScaling: bool = False):
    electronManifoldLocations = sphereTransform(electronSphereLocations)

    if sphereTransform == None:
        normals = torch.nn.functional.normalize(electronLocations, p=2, dim=-1)
    else:
        normals = SphereManifold.getManifoldNormalsFromSphereCoords(electronSphereLocations, sphereTransform)

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
    completeHessian = torch.func.vmap(callableHessian, randomness="different")(electronManifoldLocations).squeeze(dim=(1,2))
    embeddedHessians = torch.func.vmap(extractRelevantDerivatives, randomness="different")(completeHessian)

    if sphereTransform == None:
        totalCurvature = 2.
    else:
        totalCurvature = SphereManifold.getTotalCurvature(electronSphereLocations, sphereTransform)

    embeddedLaplacians = torch.sum(torch.diagonal(embeddedHessians, offset=0, dim1=-1, dim2=-2), -1).squeeze(-1)
    normalCorrection = torch.matmul(torch.matmul(normals.unsqueeze(-2), embeddedHessians), normals.unsqueeze(-1)).squeeze()
    callableGradient = torch.func.jacfwd(callableWaveFunction)
    embeddedGradients = torch.func.vmap(callableGradient, randomness="different")(electronManifoldLocations).squeeze(dim=(1,2))
    curvatureCorrection = totalCurvature * torch.matmul(normals.unsqueeze(-2), embeddedGradients.unsqueeze(-1)).squeeze()

    result = embeddedLaplacians - normalCorrection - curvatureCorrection
    return result

#This function computes the energy in terms of the Thomson problem of the electrons for every batch and then returns
#the average energy over all batches, the lowest energy of any batch and the highest energy of any batch in that order in a 3-list.
def getAvgLowHighThomsonEnergy(electronSphereLocations: torch.Tensor, sphereTransform: Callable[[torch.Tensor], torch.Tensor]):
    batchSize = electronSphereLocations.shape[0]
    batchEnergies = getThomsonEnergy(electronSphereLocations, sphereTransform)
    highEnergy = torch.max(batchEnergies).item()
    lowEnergy = torch.min(batchEnergies).item()
    avgEnergy = torch.sum(batchEnergies).item() / float(batchSize)
    medEnergy = torch.median(batchEnergies).item()
    return [avgEnergy, lowEnergy, highEnergy, medEnergy]

def getThomsonEnergy(electronSphereLocations: torch.Tensor, sphereTransform: Callable[[torch.Tensor], torch.Tensor]):
    batchSize = electronSphereLocations.shape[0]
    numElectrons = electronSphereLocations.shape[1]
    if sphereTransform == None:
        electronManifoldLocations = electronSphereLocations
    else:
        electronManifoldLocations = sphereTransform(electronSphereLocations)
    expandedElectronLocations = electronManifoldLocations.unsqueeze(2).expand(batchSize, numElectrons, numElectrons, 3).clone()
    distanceMatrix = torch.linalg.norm(expandedElectronLocations - torch.transpose(expandedElectronLocations, dim0=1, dim1=2), ord=2, dim=3)
    energyMatrix = (1. / distanceMatrix)
    helperIndex = torch.arange(numElectrons)
    energyMatrix[:, helperIndex, helperIndex] = 0.
    batchEnergies = torch.sum(energyMatrix, dim=(1,2)) / 2.
    return batchEnergies

def getAvgLowHighLocalEnergy(waveFunction: WaveFunction.MultiElectronWaveFunction, electronSphereLocations: torch.Tensor, sphereTransform: Callable[[torch.Tensor], torch.Tensor], particleMass: float = 1.0):
    batchSize = electronSphereLocations.shape[0]
    batchEnergies = computeLocalEnergy(waveFunction, electronSphereLocations, sphereTransform, particleMass)
    highEnergy = torch.max(batchEnergies).item()
    lowEnergy = torch.min(batchEnergies).item()
    avgEnergy = torch.sum(batchEnergies).item() / float(batchSize)
    medEnergy = torch.median(batchEnergies).item()
    return [avgEnergy, lowEnergy, highEnergy, medEnergy]