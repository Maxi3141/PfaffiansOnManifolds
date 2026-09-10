import numpy as np
import torch

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

# Generates a proposal for new electron positions by moving into a random direction (uniform) by a random distance (scaled chi distribution)
def generateMetropolisHastingsProposal(currentElectronPositions, sphereRadius):
    distanceScaling = 0.25
    #unitPositions = currentElectronPositions / torch.linalg.norm(currentElectronPositions, dim=-1)
    unitPositions = torch.nn.functional.normalize(currentElectronPositions, p=2, dim=-1)
    directionVectors = torch.randn(currentElectronPositions.shape)
    tangentialDirectionVectors = directionVectors - torch.matmul(directionVectors.unsqueeze(-2), unitPositions.unsqueeze(-1)).squeeze(-1) * unitPositions
    tangentialDirectionVectors = torch.nn.functional.normalize(tangentialDirectionVectors, p=2., dim=-1)
    chiDistribution = torch.distributions.chi2.Chi2(2.).expand(torch.tensor([currentElectronPositions.shape[0], currentElectronPositions.shape[1], 1]))
    randDistances = distanceScaling * torch.sqrt(chiDistribution.sample())
    proposalPositions = sphereRadius * (torch.cos(randDistances) * unitPositions + torch.sin(randDistances) * tangentialDirectionVectors)
    return proposalPositions

def conductSingleMetropolisHastingsStep(currentElectronPositions, waveFunction, sphereRadius):
    proposedElectronPositions = generateMetropolisHastingsProposal(currentElectronPositions, sphereRadius)
    densityRatio = (torch.pow(waveFunction(proposedElectronPositions), 2.) / torch.pow(waveFunction(currentElectronPositions), 2.)).squeeze(-1)
    batchSize = currentElectronPositions.shape[0]
    numElectrons = currentElectronPositions.shape[1]

    acceptQuantity = torch.rand(batchSize)
    acceptDecision = (acceptQuantity <= densityRatio).unsqueeze(-1).unsqueeze(-1).expand(batchSize, numElectrons, 3)
    possiblyNewElectronPositions = torch.where(acceptDecision, proposedElectronPositions, currentElectronPositions)

    return possiblyNewElectronPositions
        
def sampleFromWaveFunction(waveFunction, batchSize, numElectrons, sphereRadius, numMHSteps = 80):
    electronPositions = sampleUniformOnSphere(batchSize, numElectrons, sphereRadius)
    
    for _ in range(numMHSteps):
        electronPositions = conductSingleMetropolisHastingsStep(electronPositions, waveFunction, sphereRadius)
    return electronPositions

def computeModeOfWaveFunction(waveFunction, batchSize, numElectrons, sphereRadius, maxIter = 64, convergenceMultiplier: float = 0.1):
    electronPositions = sampleUniformOnSphere(batchSize, numElectrons, sphereRadius)
    waveNetworkParams = dict(waveFunction.named_parameters())

    def callableProbabilityFunction(elecPos):
        return torch.pow(torch.func.functional_call(waveFunction, waveNetworkParams, elecPos.unsqueeze(0)), 2.)

    for i in range(maxIter):
        probGradient = torch.vmap(torch.func.jacfwd(callableProbabilityFunction))(electronPositions).detach()
        probGradient = probGradient.squeeze(dim=(1,2))

        normals = torch.nn.functional.normalize(electronPositions, p=2, dim=-1)
        surfaceProjections = torch.eye(3).repeat(electronPositions.shape[0], electronPositions.shape[1], 1, 1) - torch.matmul(normals.unsqueeze(-1), normals.unsqueeze(-2))
        surfaceProbGradients = torch.matmul(surfaceProjections, probGradient.unsqueeze(-1)).squeeze(-1)

        electronPositions = electronPositions + convergenceMultiplier * surfaceProbGradients
        electronPositions = torch.nn.functional.normalize(electronPositions, p=2., dim=-1)
        #TODO: Somehow incorporate the norm of "surfaceProbGradients" to determine whether convergence has been reached.

    return electronPositions