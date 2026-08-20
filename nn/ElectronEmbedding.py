import torch

class ElectronEmbeddingNetwork(torch.nn.Module):
    def __init__(self, embeddingDim):
        super().__init__()
        self.numMLPLayers = 4
        self.embeddingDim = embeddingDim

        self.embActivation = torch.nn.functional.tanh

        #ModuleLists with two elements have the module for same-spin pairs in the first slot and and the module for different-spin pairs in the second.
        self.linear0 = torch.nn.ModuleList([torch.nn.Linear(4, self.embeddingDim, bias=False) for _ in range(2)])
        self.linear1 = torch.nn.Linear(self.embeddingDim, self.embeddingDim, bias=False)
        self.linear2 = torch.nn.Linear(self.embeddingDim, self.embeddingDim)
        self.linear3 = torch.nn.Linear(self.embeddingDim, self.embeddingDim)

        self.filterSameScaling = torch.nn.Parameter(torch.tensor(0.1))
        self.filterDiffScaling = torch.nn.Parameter(torch.tensor(0.1))

    #"electronLocations" has dimensions [NUMBER_OF_BATCHES, NUMBER_OF_ELECTRONS, 3]
    def forward(self, electronLocations: torch.Tensor, spinUpIndices: tuple[int, ...], spinDownIndices: tuple[int, ...]):
        #TODO?: Add check to ensure spinUpIndices and spinDownIndices really do partition (0,1,...,numElectrons) properly
        #... or compute spinDownIndices from spinUpIndices?
        #Update: The first NUp electrons are spin up. The rest are spin down. TODO: Replace the tuples with just the cutoff index NUp.
        batchSize = electronLocations.shape[0]
        numElectrons = electronLocations.shape[1]
        
        gRaw = self.rescaledElectronDistances(electronLocations)
        gUpUp     = self.embActivation(self.linear0[0](gRaw[:, spinUpIndices, :, :][:, :, spinUpIndices, :]))
        gUpDown   = self.embActivation(self.linear0[1](gRaw[:, spinUpIndices, :, :][:, :, spinDownIndices, :]))
        gDownUp   = self.embActivation(self.linear0[1](gRaw[:, spinDownIndices, :, :][:, :, spinUpIndices, :]))
        gDownDown = self.embActivation(self.linear0[0](gRaw[:, spinDownIndices, :, :][:, :, spinDownIndices, :]))
        gTotal = torch.cat((torch.cat((gUpUp, gUpDown), dim=-2), torch.cat((gDownDown, gDownUp), dim=-2)), dim=-3)

        pairDistances = torch.linalg.norm(self.getDistanceMatrixFromLocations(electronLocations), ord=2, dim=-1)
        filterUpUp     = self.radialFilterFunction(pairDistances[:, spinUpIndices, :][:, :, spinUpIndices], 0)
        filterUpDown   = self.radialFilterFunction(pairDistances[:, spinUpIndices, :][:, :, spinDownIndices], 1)
        filterDownUp   = self.radialFilterFunction(pairDistances[:, spinDownIndices, :][:, :, spinUpIndices], 1)
        filterDownDown = self.radialFilterFunction(pairDistances[:, spinDownIndices, :][:, :, spinDownIndices], 0)
        filterTotal = torch.cat((torch.cat((filterUpUp, filterUpDown), dim=-1), torch.cat((filterDownDown, filterDownUp), dim=-1)), dim=-2)

        electronEmbeddings = torch.sum(filterTotal.unsqueeze(-1) * gTotal, dim=-2)

        electronEmbeddings = self.embActivation(self.linear1(electronEmbeddings))
        electronEmbeddings = self.embActivation(self.linear2(electronEmbeddings))
        electronEmbeddings = self.embActivation(self.linear3(electronEmbeddings))

        return electronEmbeddings

    def getDistanceMatrixFromLocations(self, electronLocations: torch.Tensor):
        batchSize = electronLocations.shape[0]
        numElectrons = electronLocations.shape[1]
        expandedElectronLocations = electronLocations.unsqueeze(2).expand(batchSize, numElectrons, numElectrons, 3).clone()
        distanceMatrix = expandedElectronLocations - torch.transpose(expandedElectronLocations, dim0=1, dim1=2)
        return distanceMatrix

    def rescaledElectronDistances(self, electronLocations: torch.Tensor):
        distanceMatrix = self.getDistanceMatrixFromLocations(electronLocations)
        distancesNormMatrix = torch.linalg.norm(distanceMatrix, ord=2, dim=3)
        unscaledConcatMatrix = torch.cat((distanceMatrix, distancesNormMatrix.unsqueeze(-1)), -1)
        scalings = (torch.log(distancesNormMatrix + 1) / distancesNormMatrix).unsqueeze(-1)
        singularIndices = torch.arange(electronLocations.shape[1])
        scalings[:,singularIndices,singularIndices,:] = 0.
        return scalings * unscaledConcatMatrix

    def radialFilterFunction(self, x: torch.Tensor, spinInvKronecker: int):
        if spinInvKronecker == 0:
            return torch.exp(-1. * torch.pow(self.filterSameScaling * x, 2.))
        else:
            return torch.exp(-1. * torch.pow(self.filterDiffScaling * x, 2.))