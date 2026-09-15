import torch
from nn import WaveFunction
from computation import Physics
from computation import StatBasics
from decimal import Decimal

def main():
    #Attention: Always make sure that these parameters are the same as the ones that were used in training.
    numElectrons       = 4
    numSpinUpElectrons = 2
    numOrbitals        = 6
    sphereRadius       = 1.0
    particleMass       = 1e+2

    embeddingDim     = 256
    numOrbitalParams = 64
    numPfaffians     = 4
    useCuda          = False

    if useCuda:
        torch.set_default_device("cuda")

    #Mode must be "eval" to evaluate the wave function at given points...
    #... or "sample" to sample numElectron points,
    #... or "mode" to compute the Thomson energy of the modes of the distribution.
    #... or "benchmark"
    mode = "mode"

    #Only relevant when mode == "sample" or mode == "mode"
    batchSize = 128

    #Only relevant when mode is set to "eval"
    #unnormedEvaluationPoints = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]])
    
    #unnormedEvaluationPoints = torch.tensor([[[-0.0753,  0.9267, -0.3681],[-0.5428, -0.3907,  0.7434],[-0.7392, -0.6146, -0.2754],[0.8889, -0.2052, -0.4095]]])
    #unnormedEvaluationPoints = torch.tensor([[[-1.0, 0.01, 0.0], [-1.0, 0.0, 0.0]]])
    unnormedEvaluationPoints = torch.randn(32, 4, 3)

    print("--- NeuralPfaffians on Manifolds ---")
    print(f"Simulating {numElectrons} electrons ({numSpinUpElectrons} spin up, {(numElectrons - numSpinUpElectrons)}, spin down) in {numOrbitals} orbitals.")
    print(f"Make sure that a pretrained pth-File for this configuration is available in \"/saves\".")

    if mode != "eval" and mode != "sample" and mode != "mode" and mode != "benchmark":
        raise Exception("Parameter \"mode\" is invalid. Must be \"eval\", \"sample\", \"mode\" or \"benchmark\".")

    if mode == eval:
        if len(unnormedEvaluationPoints.shape) != 3:
            raise Exception("\"unnormedEvaluationPoints\"-Tensor is invalid. It must have shape [BATCH SIZE , NUMBER OF ELECTRONS , 3].")
        if unnormedEvaluationPoints.shape[1] != numElectrons:
            raise Exception(f"\"unnormedEvaluationPoints\"-Tensor is invalid. It apparently has {unnormedEvaluationPoints.shape[1]} electrons but \"numElectrons\"-parameter is set to {numElectrons}.")
    normedEvaluationPoints = sphereRadius * torch.nn.functional.normalize(unnormedEvaluationPoints, p=2., dim=-1)

    waveNetwork = WaveFunction.MultiElectronWaveFunction(numElectrons, numOrbitals, numSpinUpElectrons, sphereRadius, particleMass, embeddingDim, numOrbitalParams, numPfaffians)
    print(f"Created wave function Pfaffians with {sum(p.numel() for p in waveNetwork.parameters())} parameters in total.")
    strMass = '%.2E' % Decimal(particleMass)
    waveNetwork.load_state_dict(torch.load(f"./saves/ManifoldPfaffian_{numElectrons}E_{numOrbitals}O_{numSpinUpElectrons}Up_M{strMass}.pth"))
    print("Loaded weights from save file.")

    if mode == "eval":
        result = waveNetwork(normedEvaluationPoints)
        print("Evaluated wave function. Result:")
        print(result)
    if mode == "sample":
        result = StatBasics.sampleFromWaveFunction(waveNetwork, batchSize, numElectrons, sphereRadius, 64)
        print(f"Sampled {numElectrons} electrons {batchSize} times. Result:")
        print(result)
        waveAtLocations = waveNetwork(result)
        print(f"... with wave function values of:")
        print(waveAtLocations)
    if mode == "mode":
        result = StatBasics.computeModeOfWaveFunction(waveNetwork, batchSize, numElectrons, sphereRadius, 128, 1.0)
        modeThomsonEnergy = Physics.getAvgLowHighThomsonEnergy(result)
        print(f"Estimated mode of wave function. Thomson energies over batch: Low = {modeThomsonEnergy[1]} ; Avg = {modeThomsonEnergy[0]} ; High = {modeThomsonEnergy[2]}")
    if mode == "benchmark":
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as prof:
            output = waveNetwork(normedEvaluationPoints)

        print(
            prof.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=30
            )
        )

if __name__=="__main__":
    main()