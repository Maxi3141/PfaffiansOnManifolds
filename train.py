import torch
from nn import WaveFunction
from computation import Physics
from computation import StatBasics
from decimal import Decimal

def main():
    numElectrons       = 4
    numSpinUpElectrons = 2
    numOrbitals        = 6
    sphereRadius       = 1.0
    particleMass       = 1e+2

    batchSize          = 8
    embeddingDim       = 256
    numOrbitalParams   = 64
    numPfaffians       = 4

    resumeTraining         = True
    computeEnergyInterval  = 10
    computeThomsonInterval = 1000
    computeModeInterval    = 1000
    maxTrainingIter        = 800
    saveInterval           = 10
    useCuda                = False
    optimizerMomentum      = 0.99
    optimizerDamping       = 0.000

    print("--- NeuralPfaffians on Manifolds ---")
    print(f"Simulating {numElectrons} electrons ({numSpinUpElectrons} spin up, {(numElectrons - numSpinUpElectrons)}, spin down) in {numOrbitals} orbitals.")

    if useCuda:
        torch.set_default_device("cuda")

    strMass = '%.2E' % Decimal(particleMass)

    waveNetwork = WaveFunction.MultiElectronWaveFunction(numElectrons, numOrbitals, numSpinUpElectrons, sphereRadius, particleMass, embeddingDim, numOrbitalParams, numPfaffians)
    print(f"Created wave function Pfaffians with {sum(p.numel() for p in waveNetwork.parameters())} parameters in total.")
    if resumeTraining:
        waveNetwork.load_state_dict(torch.load(f"./saves/ManifoldPfaffian_{numElectrons}E_{numOrbitals}O_{numSpinUpElectrons}Up_M{strMass}.pth"))

    #TODO: Do something with the energy histories.
    localEnergyHistory   = []
    thomsonEnergyHistory = []

    gradientMemory = [torch.zeros_like(subParameter) for subParameter in waveNetwork.getLogGradient(torch.zeros(1, numElectrons, 3))[0]]
    for iter in range(maxTrainingIter):
        print(f"iteration {iter}...")

        electronLocations = StatBasics.sampleFromWaveFunction(waveNetwork, batchSize, numElectrons, sphereRadius)

        learningRate = 0.08 * (1.0 + float(iter + 3000) * 1e-4)**-1
        currentGradient = Physics.estimateVMCGradient(waveNetwork, batchSize, electronLocations)
        gradientMemory = [optimizerMomentum * gradientMemory[i] + (1. - optimizerMomentum) * currentGradient[i].detach().clone() for i in range(len(gradientMemory))]
        if iter == 0:
            gradientMemory = currentGradient
        waveNetwork.updateWeights(gradientMemory, (1. - optimizerDamping) * learningRate)
        print(f"   -> Updated weights using VMC gradient.")

        if iter % saveInterval == 0:
            torch.save(waveNetwork.state_dict(), f"./saves/ManifoldPfaffian_{numElectrons}E_{numOrbitals}O_{numSpinUpElectrons}Up_M{strMass}.pth")
            print(f"   -> Saved weights.")

        if iter % computeEnergyInterval == 0:
            localEnergyData = Physics.getAvgLowHighLocalEnergy(waveNetwork, electronLocations, sphereRadius, particleMass)
            localEnergyHistory.append(localEnergyData)
            print(f"   -> Local energy in batch of {batchSize}: Low = {localEnergyData[1]} ; Avg = {localEnergyData[0]} ; High = {localEnergyData[2]}")

        if iter % computeThomsonInterval == 0:
            thomsonEnergyData = Physics.getAvgLowHighThomsonEnergy(electronLocations)
            thomsonEnergyHistory.append(thomsonEnergyData)
            print(f"   -> Thomson energy in batch of {batchSize}: Low = {thomsonEnergyData[1]} ; Avg = {thomsonEnergyData[0]} ; High = {thomsonEnergyData[2]}")

        if iter % computeModeInterval == 0:
            modePositions = StatBasics.computeModeOfWaveFunction(waveNetwork, batchSize, numElectrons, sphereRadius)
            modeThomsonEnergy = Physics.getAvgLowHighThomsonEnergy(modePositions)
            print(f"   -> Thomson energy of modes: Low = {modeThomsonEnergy[1]} ; Avg = {modeThomsonEnergy[0]} ; High = {modeThomsonEnergy[2]}")

    print("Avg. energies:")
    print([record[0] for record in localEnergyHistory])

if __name__=="__main__":
    main()