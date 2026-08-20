import numpy as np
import torch

def sampleUniformOnSphereNumpy(r = 1):
    theta = 2 * np.pi * np.random.uniform()
    phi = np.acos(2 * np.random.uniform() - 1)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z])

def sampleUniformOnSphere(r = 1.):
    theta = 2. * torch.pi * torch.rand(1)
    phi = torch.acos(2. * torch.rand(1) - 1.)
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)
    return torch.cat([x, y, z])

# Generates a proposal for new electron positions by moving into a random direction (uniform) by a random distance (scaled chi distribution)
def generateMetropolisHastingsProposal(currentElectronPositions, sphereRadius):
    distanceScaling = 0.1
    #unitPositions = currentElectronPositions / torch.linalg.norm(currentElectronPositions, dim=-1)
    unitPositions = torch.nn.functional.normalize(currentElectronPositions, p=2, dim=-1)
    directionVectors = torch.randn(currentElectronPositions.shape)
    tangentialDirectionVectors = directionVectors - torch.matmul(directionVectors.unsqueeze(-2), unitPositions.unsqueeze(-1)).squeeze(-1) * unitPositions
    chiDistribution = torch.distributions.chi2.Chi2(2.).expand(torch.tensor([currentElectronPositions.shape[0], currentElectronPositions.shape[1], 1]))
    randDistances = distanceScaling * torch.sqrt(chiDistribution.sample())
    proposalPositions = sphereRadius * (torch.cos(randDistances) * unitPositions + torch.sin(randDistances) * tangentialDirectionVectors)
    return proposalPositions

def conductMetropolisHastingsStep(currentElectronPositions, waveFunction, sphereRadius):
    proposedElectronPositions = generateMetropolisHastingsProposal(currentElectronPositions, sphereRadius)
    densityRatio = waveFunction(proposedElectronPositions)**2 / waveFunction(currentElectronPositions)**2
    if densityRatio >= 1:
        return proposedElectronPositions
    else:
        acceptQuantity = np.random.uniform()
        if acceptQuantity <= densityRatio:
            return proposedElectronPositions
        else:
            return currentElectronPositions
        
def sampleFromWaveFunction(waveFunction, numElectrons, sphereRadius, numMHSteps = 10):
    electronPositions = torch.stack([sampleUniformOnSphere(sphereRadius) for _ in range(numElectrons)]).unsqueeze(0)
    for _ in range(numMHSteps):
        electronPositions = conductMetropolisHastingsStep(electronPositions, waveFunction, sphereRadius)
    return electronPositions

def sampleBatchFromWaveFunction(batchSize, waveFunction, numElectrons, sphereRadius, numMHSteps = 10):
    return torch.cat([sampleFromWaveFunction(waveFunction, numElectrons, sphereRadius, numMHSteps) for _ in range(batchSize)])