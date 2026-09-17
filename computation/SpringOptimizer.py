import torch
from computation import Physics
from nn import WaveFunction

#TODO: Invert the system below more elegantly. The matrix to invert can become non positive-definit and then everything breaks.
#... skipping all the optimization steps where this happens is bad and should not happen with some future solution.
def getSpringOptimizerGradient(waveFunction:       WaveFunction.MultiElectronWaveFunction, 
                               electronLocations:  torch.Tensor, 
                               prevSpringGradient: list[torch.Tensor], 
                               damping:            float, 
                               momentum:           float, 
                               stabilizingConst:   float = 1.0):
    batchSize = electronLocations.shape[0]
    currentRawGradientTensors = waveFunction.getLogGradient(electronLocations)
    currentRawGradient = torch.cat([torch.flatten(subTensor.unsqueeze(-1), start_dim=1) for subTensor in currentRawGradientTensors], dim=1)
    stabilizingHelper = torch.ones(batchSize, batchSize, dtype=torch.float64) / float(batchSize) 

    #for subIndex, subParameterGradients in enumerate(currentRawGradient):
    bigO = 1. / float(batchSize)**0.5 * currentRawGradient
    epsE = -1. / float(batchSize)**0.5 * Physics.computeLocalEnergy(waveFunction, electronLocations, waveFunction.sphereRadius, waveFunction.particleMass)

    bigOBar = bigO - torch.sum(bigO, dim=0) / float(batchSize)
    epsEBar = epsE - torch.sum(epsE, dim=0) / float(batchSize)  

    bigOBar64 = bigOBar.to(dtype=torch.float64)
    invTerm64     = torch.matmul(bigOBar64, torch.transpose(bigOBar64, dim0=0, dim1=1)) + damping * torch.eye(batchSize, dtype=torch.float64) + stabilizingConst * stabilizingHelper
    invTermL64, _ = torch.linalg.cholesky_ex(invTerm64, check_errors=False)

    if invTermL64.diagonal()[torch.where(invTermL64.diagonal() <= 1e-10)].shape[0] != 0:
        print(f"WARNING: Spring optimizer could not invert matrix.")
        return torch.zeros(1).detach(), False

    invTerm64    = torch.cholesky_inverse(invTermL64)
    invTerm      = invTerm64.to(dtype=torch.float32)
    energyTerm   = epsEBar - momentum * torch.matmul(bigOBar, prevSpringGradient)
    momentumTerm = momentum * prevSpringGradient
    result = torch.transpose(bigOBar, dim0=0, dim1=1) @ invTerm @ energyTerm + momentumTerm

    if result[torch.where(torch.isnan(result))].shape[0] != 0:
        print(f"WARNING: Spring optimizer somehow came up with tensor containing NaN. Skipping this optimization step.")
        return torch.zeros(1).detach(), False

    return result.detach(), True