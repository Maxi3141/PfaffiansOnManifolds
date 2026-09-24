import torch
from nn import WaveFunction
from computation import Physics
from computation import StatBasics
from computation import SpringOptimizer
from decimal import Decimal

def main():
    numElectrons       = 4
    numSpinUpElectrons = 2
    numOrbitals        = 6
    sphereRadius       = 1.0
    particleMass       = 1e+0

    batchSize          = 64
    embeddingDim       = 256
    numOrbitalParams   = 64
    numPfaffians       = 4

    resumeTraining         = False
    computeEnergyInterval  = 10
    computeThomsonInterval = 20
    computeModeInterval    = 100
    maxTrainingIter        = 1001
    saveInterval           = 10
    permaSaveInterval      = 500
    useCuda                = True
    optimizerMomentum      = 0.99
    optimizerDamping       = 0.001

    print("--- NeuralPfaffians on Manifolds ---")
    print(f"Simulating {numElectrons} electrons ({numSpinUpElectrons} spin up, {(numElectrons - numSpinUpElectrons)}, spin down) in {numOrbitals} orbitals.")

    if numElectrons % 2 != 0:
        raise Exception("Odd number of electrons not yet supported!")

    if useCuda:
        print("WARNING: CUDA is not yet fully supported and results could be wrong!")
        torch.set_default_device("cuda")

    strMass = '%.2E' % Decimal(particleMass)

    waveNetwork = WaveFunction.MultiElectronWaveFunction(numElectrons, numOrbitals, numSpinUpElectrons, sphereRadius, particleMass, embeddingDim, numOrbitalParams, numPfaffians)
    print(f"Created wave function Pfaffians with {sum(p.numel() for p in waveNetwork.parameters())} parameters in total.")
    if resumeTraining:
        waveNetwork.load_state_dict(torch.load(f"./saves/ManifoldPfaffian_{numElectrons}E_{numOrbitals}O_{numSpinUpElectrons}Up_M{strMass}.pth"))

    localEnergyHistory   = []
    thomsonEnergyHistory = []
    modeHistory          = []

    numSkippedTrainingIters = 0

    prevGradient = torch.zeros_like(torch.nn.utils.parameters_to_vector(waveNetwork.getLogGradient(torch.zeros(1, numElectrons, 3))))
    tensorFormattedGradient = [torch.zeros_like(subParam) for subParam in waveNetwork.parameters()]
    iterStart = 0
    for iter in range(iterStart, maxTrainingIter):
        print(f"iteration {iter}...")

        electronLocations = StatBasics.sampleFromWaveFunction(waveNetwork, batchSize, numElectrons, sphereRadius)

        learningRate = 0.08 * (1.0 + float(iter) * 1e-4)**-1
        springGradient, springSuccess = SpringOptimizer.getSpringOptimizerGradient(waveNetwork, electronLocations, prevGradient, optimizerDamping, optimizerMomentum)
        if not springSuccess:
            numSkippedTrainingIters += 1
            print(f" Optimizer step computation failed. {numSkippedTrainingIters} iterations were now skipped in total.")
            continue

        torch.nn.utils.vector_to_parameters(springGradient, tensorFormattedGradient)
        prevGradient = springGradient
        waveNetwork.updateWeights(tensorFormattedGradient, learningRate)
        print(f"   -> Updated weights using Spring gradient.")

        if iter % saveInterval == 0:
            torch.save(waveNetwork.state_dict(), f"./saves/ManifoldPfaffian_{numElectrons}E_{numOrbitals}O_{numSpinUpElectrons}Up_M{strMass}.pth")
            print(f"   -> Saved weights.")

        if iter % permaSaveInterval == 0 and iter != 0:
            torch.save(waveNetwork.state_dict(), f"./saves/ManifoldPfaffian_{numElectrons}E_{numOrbitals}O_{numSpinUpElectrons}Up_M{strMass}_{iter}.pth")
            print(f"   -> Saved weights for current iteration.")

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
            modeHistory.append(modeThomsonEnergy)
            print(f"   -> Thomson energy of modes: Low = {modeThomsonEnergy[1]} ; Avg = {modeThomsonEnergy[0]} ; High = {modeThomsonEnergy[2]}")

    with open("saves/lastSessionData.log", "w") as f:
        f.write(f"numElectrons = {numElectrons} , up = {numSpinUpElectrons} , numOrbitals = {numOrbitals} , batchSize = {batchSize}\n")
        f.write("Average energy history:\n")
        f.write(" , ".join([str(record[0]) for record in localEnergyHistory]) + "\n")
        f.write("Average Thomson energy history:\n")
        f.write(" , ".join([str(record[0]) for record in thomsonEnergyHistory]) + "\n")
        f.write("Average Thomson energy of modes history:\n")
        f.write(" , ".join([str(record[0]) for record in modeHistory]) + "\n")
        f.close()

if __name__=="__main__":
    main()