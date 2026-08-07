import numpy as np

def sampleUniformOnSphere(r = 1):
    theta = 2 * np.pi * np.random.uniform()
    phi = np.acos(2 * np.random.uniform() - 1)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z])

# Generates a proposal for new electron positions by moving into a random direction (uniform) by a random distance (scaled chi distribution)
def generateMetropolisHastingsProposal(currentElectronPositions, sphereRadius):
    proposalPositions = []
    distanceScaling = 0.1
    for position in currentElectronPositions:
        unitPosition = position / np.linalg.norm(position)
        directionVector = np.random.standard_normal(3)
        directionVector = directionVector - np.dot(directionVector, unitPosition) * unitPosition
        directionVector = directionVector / np.linalg.norm(directionVector)
        distance = distanceScaling * np.random.chisquare(2)**0.5
        newPosition = sphereRadius * (np.cos(distance) * unitPosition + np.sin(distance) * directionVector)
        proposalPositions.append(newPosition)

def conductMetropolisHastingsStep(currentElectronPositions, waveFunction, sphereRadius):
    proposedElectronPositions = generateMetropolisHastingsProposal(currentElectronPositions, sphereRadius)
    densityRatio = waveFunction(proposedElectronPositions)**2 / waveFunction(currentElectronPositions)**2
    if densityRatio >= 1:
        return proposedElectronPositions
    else:
        accept_quantity = np.random.uniform()
        if accept_quantity <= densityRatio:
            return proposedElectronPositions
        else:
            return currentElectronPositions
        
def sampleFromWaveFunction(waveFunction, numElectrons, sphereRadius, numMHSteps = 10):
    electronPositions = [sampleUniformOnSphere(sphereRadius) for _ in range(numElectrons)]
    for _ in range(numMHSteps):
        electronPositions = conductMetropolisHastingsStep(electronPositions, waveFunction, sphereRadius)
    return electronPositions
