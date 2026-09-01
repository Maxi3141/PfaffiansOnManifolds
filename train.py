import torch
from nn import WaveFunction
from computation import Physics
from computation import StatBasics
from decimal import Decimal

def main():
    numElectrons       = 2
    numSpinUpElectrons = 1
    numOrbitals        = 3
    sphereRadius       = 1.0
    particleMass       = 1e-1

    resumeTraining         = True
    computeEnergyInterval  = 10
    computeThomsonInterval = 10
    maxTrainingIter        = 10000
    saveInterval           = 10
    batchSize              = 16
    embeddingDim           = 128
    numOrbitalParams       = 32
    useCuda                = False

    print("--- NeuralPfaffians on Manifolds ---")
    print(f"Simulating {numElectrons} electrons ({numSpinUpElectrons} spin up, {(numElectrons - numSpinUpElectrons)}, spin down) in {numOrbitals} orbitals.")

    if useCuda:
        torch.set_default_device("cuda")

    strMass = '%.2E' % Decimal(particleMass)

    waveNetwork = WaveFunction.MultiElectronWaveFunction(numElectrons, numOrbitals, numSpinUpElectrons, sphereRadius, particleMass, embeddingDim, numOrbitalParams)
    print(f"Created wave function Pfaffians with {sum(p.numel() for p in waveNetwork.parameters())} parameters in total.")
    if resumeTraining:
        waveNetwork.load_state_dict(torch.load(f"./saves/ManifoldPfaffian_{numElectrons}E_{numOrbitals}O_{numSpinUpElectrons}Up_M{strMass}.pth"))

    #TODO: Do something with the energy histories.
    localEnergyHistory   = []
    thomsonEnergyHistory = []

    for iter in range(maxTrainingIter):
        print(f"iteration {iter}...")

        electronLocations = StatBasics.sampleFromWaveFunction(waveNetwork, batchSize, numElectrons, sphereRadius)

        #learningRate = 0.08 * (1.0 + float(iter) * 1e-4)**-1
        learningRate = 0.08 * (1.0 + float(iter) * 1e-4)**-1
        parameterGradient = Physics.estimateVMCGradient(waveNetwork, batchSize, electronLocations)
        waveNetwork.updateWeights(parameterGradient, learningRate)
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

if __name__=="__main__":
    main()