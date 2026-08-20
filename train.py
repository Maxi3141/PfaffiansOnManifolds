import torch
from nn import WaveFunction
from computation import Physics
from computation import StatBasics

def main():
    numElectrons       = 6
    numSpinUpElectrons = 4
    numOrbitals        = 5
    sphereRadius       = 1.0
    particleMass       = 1.0

    resumeTraining         = False
    computeEnergyInterval  = 10
    computeThomsonInterval = 10
    maxTrainingIter        = 100000
    saveInterval           = 100
    batchSize              = 16
    embeddingDim           = 128
    useCuda                = True

    print("--- NeuralPfaffians on Manifolds ---")
    print(f"Simulating {numElectrons} electrons ({numSpinUpElectrons} spin up, {(numElectrons - numSpinUpElectrons)}, spin down) in {numOrbitals} orbitals.")

    if useCuda:
        torch.set_default_device("cuda")

    waveNetwork = WaveFunction.MultiElectronWaveFunction(numElectrons, numOrbitals, numSpinUpElectrons, sphereRadius, particleMass, embeddingDim)
    waveNetwork.to("cuda")
    print(f"Created wave function Pfaffians with {sum(p.numel() for p in waveNetwork.parameters())} parameters in total.")
    if resumeTraining:
        waveNetwork.load_state_dict(torch.load(f"./saves/ManifoldPfaffian_{numElectrons}E_{numOrbitals}O_{numSpinUpElectrons}Up.pth"))

    #TODO: Do something with the energy histories.
    localEnergyHistory   = []
    thomsonEnergyHistory = []

    for iter in range(maxTrainingIter):
        print(f"iteration {iter}...")

        learningRate = 0.02 * (1.0 + float(iter) * 1e-4)**-1
        parameterGradient = Physics.estimateVMCGradient(waveNetwork, batchSize)
        waveNetwork.updateWeights(parameterGradient, learningRate)
        print(f"   -> Updated weights using VMC gradient.")

        if iter % saveInterval == 0:
            torch.save(waveNetwork.state_dict(), f"./saves/ManifoldPfaffian_{numElectrons}E_{numOrbitals}O_{numSpinUpElectrons}Up.pth")
            print(f"   -> Saved weights.")

        if iter % computeEnergyInterval == 0:
            electronLocations = StatBasics.sampleBatchFromWaveFunction(batchSize, waveNetwork, numElectrons, sphereRadius)
            localEnergyData = Physics.getAvgLowHighLocalEnergy(waveNetwork, electronLocations, sphereRadius, particleMass)
            localEnergyHistory.append(localEnergyData)
            print(f"   -> Local energy in batch of {batchSize}: Low = {localEnergyData[1]} ; Avg = {localEnergyData[0]} ; High = {localEnergyData[2]}")

        if iter % computeThomsonInterval == 0:
            electronLocations = StatBasics.sampleBatchFromWaveFunction(batchSize, waveNetwork, numElectrons, sphereRadius)
            thomsonEnergyData = Physics.getAvgLowHighThomsonEnergy(electronLocations)
            thomsonEnergyHistory.append(thomsonEnergyData)
            print(f"   -> Thomson energy in batch of {batchSize}: Low = {thomsonEnergyData[1]} ; Avg = {thomsonEnergyData[0]} ; High = {thomsonEnergyData[2]}")

if __name__=="__main__":
    main()