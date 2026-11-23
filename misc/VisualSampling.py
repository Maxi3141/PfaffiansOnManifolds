import torch
from nn import WaveFunction
from computation import StatBasics
from computation import SphereManifold
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

def main():
    numElectrons       = 2
    numSpinUpElectrons = 1
    numOrbitals        = 3
    particleMass       = 1e+0
    sphereTransform    = SphereManifold.eggTransform

    embeddingDim     = 256
    numOrbitalParams = 64
    numPfaffians     = 4

    batchSize = 1024

    print("--- NeuralPfaffians on Manifolds ---")
    print(f"Sampling {numElectrons} electrons ({numSpinUpElectrons} spin up, {(numElectrons - numSpinUpElectrons)}, spin down) in {numOrbitals} orbitals {batchSize} times.")
    print(f"Make sure that a pretrained pth-File for this configuration is available in \"/saves\".")

    waveNetwork = WaveFunction.MultiElectronWaveFunction(numElectrons, numOrbitals, numSpinUpElectrons, particleMass, embeddingDim, numOrbitalParams, numPfaffians)
    print(f"Created wave function Pfaffians with {sum(p.numel() for p in waveNetwork.parameters())} parameters in total.")
    strMass = '%.2E' % Decimal(particleMass)
    waveNetwork.load_state_dict(torch.load(f"./saves/ManifoldPfaffian_{numElectrons}E_{numOrbitals}O_{numSpinUpElectrons}Up_M{strMass}_{SphereManifold.transformationName(sphereTransform)}.pth"))
    print("Loaded weights from save file.")

    sampledSphereLocations = StatBasics.sampleFromWaveFunctionOnReferenceSphere(waveNetwork, batchSize, numElectrons, sphereTransform)
    sampledManifoldLocations = sampledSphereLocations
    if sphereTransform != None:
        sampledManifoldLocations = sphereTransform(sampledSphereLocations)

    sampledManifoldLocations = torch.flatten(sampledManifoldLocations, start_dim=0, end_dim=-2)

    numpyLocations = sampledManifoldLocations.detach().cpu().numpy()

    print(numpyLocations)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(numpyLocations[:,0], numpyLocations[:,1], numpyLocations[:,2])
    ax.set_aspect("equal")
    plt.show()

if __name__=="__main__":
    main()