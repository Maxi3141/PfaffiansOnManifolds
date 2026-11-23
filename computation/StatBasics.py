import numpy as np
import torch
import math
from computation import SphereManifold

def sampleUniformOnSphereNumpy(r = 1):
    theta = 2 * np.pi * np.random.uniform()
    phi = np.acos(2 * np.random.uniform() - 1)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z])

def sampleUniformOnSphere(batchSize, numElectrons, r = 1.):
    theta = 2. * torch.pi * torch.rand(batchSize, numElectrons)
    phi = torch.acos(2. * torch.rand(batchSize, numElectrons) - 1.)
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)
    return torch.stack([x, y, z], dim=2)

def samplePreuniformOnManifold(batchSize, numElectrons, sphereDiffeomorphism):
    sphereSample = sampleUniformOnSphere(batchSize, numElectrons)
    return sphereDiffeomorphism(sphereSample)

# Generates a proposal for new electron positions by moving into a random direction (uniform) by a random distance (scaled chi distribution)
def generateMetropolisHastingsProposal(currentElectronPositions):
    distanceScaling = 0.25
    unitPositions = torch.nn.functional.normalize(currentElectronPositions, p=2, dim=-1)
    directionVectors = torch.randn(currentElectronPositions.shape)
    tangentialDirectionVectors = directionVectors - torch.matmul(directionVectors.unsqueeze(-2), unitPositions.unsqueeze(-1)).squeeze(-1) * unitPositions
    tangentialDirectionVectors = torch.nn.functional.normalize(tangentialDirectionVectors, p=2., dim=-1)
    chiDistribution = torch.distributions.chi2.Chi2(2.).expand(torch.tensor([currentElectronPositions.shape[0], currentElectronPositions.shape[1], 1]))
    randDistances = distanceScaling * torch.sqrt(chiDistribution.sample())
    proposalPositions = torch.cos(randDistances) * unitPositions + torch.sin(randDistances) * tangentialDirectionVectors
    return proposalPositions

def conductSingleMetropolisHastingsStep(currentElectronSpherePositions, waveFunction, sphereTransform):
    proposedElectronSpherePositions = generateMetropolisHastingsProposal(currentElectronSpherePositions)
    currentElectronManifoldPositions = currentElectronSpherePositions
    proposedElectronManifoldPositions = proposedElectronSpherePositions
    if sphereTransform != None:
        currentElectronManifoldPositions = sphereTransform(currentElectronSpherePositions)
        proposedElectronManifoldPositions = sphereTransform(proposedElectronSpherePositions)
    densityRatio = (torch.pow(waveFunction(proposedElectronManifoldPositions), 2.) / torch.pow(waveFunction(currentElectronManifoldPositions), 2.)).squeeze(-1)
    batchSize = currentElectronSpherePositions.shape[0]
    numElectrons = currentElectronSpherePositions.shape[1]

    acceptQuantity = torch.rand(batchSize)
    acceptDecision = (acceptQuantity <= densityRatio).unsqueeze(-1).unsqueeze(-1).expand(batchSize, numElectrons, 3)
    possiblyNewElectronSpherePositions = torch.where(acceptDecision, proposedElectronSpherePositions, currentElectronSpherePositions)

    return possiblyNewElectronSpherePositions
        
def sampleFromWaveFunctionOnReferenceSphere(waveFunction, batchSize, numElectrons, sphereTransform, numMHSteps = 32):
    electronSpherePositions = sampleUniformOnSphere(batchSize, numElectrons)
    
    for _ in range(numMHSteps):
        electronSpherePositions = conductSingleMetropolisHastingsStep(electronSpherePositions, waveFunction, sphereTransform)
    return electronSpherePositions

def computeModeOfWaveFunctionOnSphere(waveFunction, batchSize, numElectrons, sphereTransform, maxIter = 64, convergenceMultiplier: float = 0.1, filterNANs: bool = True):
    electronSpherePositions = sampleUniformOnSphere(batchSize, numElectrons)
    waveNetworkParams = dict(waveFunction.named_parameters())

    def callableProbabilityFunction(elecPos):
        return torch.pow(torch.func.functional_call(waveFunction, waveNetworkParams, elecPos.unsqueeze(0)), 2.)

    for _ in range(maxIter):
        electronManifoldPositions = electronSpherePositions
        if sphereTransform != None:
            electronManifoldPositions = sphereTransform(electronSpherePositions)
        probGradient = torch.vmap(torch.func.jacfwd(callableProbabilityFunction))(electronManifoldPositions).detach()
        probGradient = probGradient.squeeze(dim=(1,2))

        if sphereTransform == None:
            normals = torch.nn.functional.normalize(electronSpherePositions, p=2, dim=-1)
        else:
            normals = SphereManifold.getManifoldNormalsFromSphereCoords(electronSpherePositions, sphereTransform)
        surfaceProjections = torch.eye(3).repeat(electronSpherePositions.shape[0], electronSpherePositions.shape[1], 1, 1) - torch.matmul(normals.unsqueeze(-1), normals.unsqueeze(-2))
        #surfaceProbGradients = torch.matmul(surfaceProjections, torch.clamp(probGradient, min=-1e-3, max=1e-3).unsqueeze(-1)).squeeze(-1)
        surfaceProbGradients = torch.matmul(surfaceProjections, probGradient.unsqueeze(-1)).squeeze(-1)

        convergenceScaling = convergenceMultiplier
        electronSpherePositions = electronSpherePositions + convergenceScaling * surfaceProbGradients
        electronSpherePositions = torch.nn.functional.normalize(electronSpherePositions, p=2., dim=-1)
        #TODO: Somehow incorporate the norm of "surfaceProbGradients" to determine whether convergence has been reached.

    if filterNANs:
        nanMask = torch.any(electronSpherePositions.isnan(), dim=(1,2))
        electronSpherePositions = electronSpherePositions[~nanMask]

    return electronSpherePositions