import torch
from nn import WaveFunction
from computation import Physics
from computation import StatBasics

def main():
    #Attention: Always make sure that these parameters are the same as the ones that were used in training.
    numElectrons       = 6
    numSpinUpElectrons = 4
    numOrbitals        = 5
    sphereRadius       = 1.0
    particleMass       = 1.0

    embeddingDim = 128
    useCuda      = False

    #Mode must be "eval" to evaluate the wave function at given points...
    #... or "sample" to sample numElectron points.
    mode = "sample"

    #Only relevant when mode == "sample"
    batchSize = 10

    #Only relevant when mode is set to "eval"
    unnormedEvaluationPoints = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]])

    print("--- NeuralPfaffians on Manifolds ---")
    print(f"Simulating {numElectrons} electrons ({numSpinUpElectrons} spin up, {(numElectrons - numSpinUpElectrons)}, spin down) in {numOrbitals} orbitals.")
    print(f"Make sure that a pretrained pth-File for this configuration is available in \"/saves\".")

    if mode != "eval" and mode != "sample":
        raise Exception("Parameter \"mode\" is invalid. Must be \"eval\" or \"sample\".")

    if useCuda:
        torch.set_default_device("cuda")

    if mode == eval:
        if len(unnormedEvaluationPoints.shape) != 3:
            raise Exception("\"unnormedEvaluationPoints\"-Tensor is invalid. It must have shape [BATCH SIZE , NUMBER OF ELECTRONS , 3].")
        if unnormedEvaluationPoints.shape[1] != numElectrons:
            raise Exception(f"\"unnormedEvaluationPoints\"-Tensor is invalid. It apparently has {unnormedEvaluationPoints.shape[1]} electrons but \"numElectrons\"-parameter is set to {numElectrons}.")
        normedEvaluationPoints = sphereRadius * torch.nn.functional.normalize(unnormedEvaluationPoints, p=2., dim=-1)

    waveNetwork = WaveFunction.MultiElectronWaveFunction(numElectrons, numOrbitals, numSpinUpElectrons, sphereRadius, particleMass, embeddingDim)
    print(f"Created wave function Pfaffians with {sum(p.numel() for p in waveNetwork.parameters())} parameters in total.")
    waveNetwork.load_state_dict(torch.load(f"./saves/ManifoldPfaffian_{numElectrons}E_{numOrbitals}O_{numSpinUpElectrons}Up.pth"))
    print("Loaded weights from save file.")

    if mode == "eval":
        result = waveNetwork(unnormedEvaluationPoints)
        print("Evaluated wave function. Result:")
        print(result)
    if mode == "sample":
        result = StatBasics.sampleFromWaveFunction(waveNetwork, batchSize, numElectrons, sphereRadius, 64)
        print(f"Sampled {numElectrons} electrons {batchSize} times. Result:")
        print(result)

if __name__=="__main__":
    main()