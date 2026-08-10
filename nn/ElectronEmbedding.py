import torch

class ElectronEmbedding(torch.nn.Module):
    def __init__(self, embeddingDim):
        self.numMLPLayers = 4
        self.embeddingDim = embeddingDim
        self.filterHiddenDim = 16

        self.siluActivation = torch.nn.functional.silu

        #ModuleLists with two elements have the module for same-spin pairs in the first slot and and the module for different-spin pairs in the second.
        self.linear0 = torch.nn.ModuleList([torch.nn.Linear(4, self.embeddingDim, bias=False) for _ in range(2)])
        self.linear1 = torch.nn.Linear(self.embeddingDim, self.embeddingDim, bias=False)
        self.linear2 = torch.nn.Linear(self.embeddingDim, self.embeddingDim)
        self.linear3 = torch.nn.Linear(self.embeddingDim, self.embeddingDim)

        self.filterScaling  = torch.nn.ModuleList([torch.nn.Linear(1, self.filterHiddenDim, bias=False) for _ in range(2)])
        self.filterLinear00 = torch.nn.ModuleList([torch.nn.Linear(self.filterHiddenDim, self.filterHiddenDim, bias=False) for _ in range(2)])
        self.filterLinear10 = torch.nn.ModuleList([torch.nn.Linear(1, self.filterHiddenDim) for _ in range(2)])
        self.filterLinear11 = torch.nn.ModuleList([torch.nn.Linear(self.filterHiddenDim, self.filterHiddenDim) for _ in range(2)])
        self.filterLinear2  = torch.nn.ModuleList([torch.nn.Linear(self.filterHiddenDim, self.embeddingDim, bias=False) for _ in range(2)])

    #"electronLocations" has dimensions [NUMBER_OF_BATCHES, NUMBER_OF_ELECTRONS, 3]
    def forward(self, electronLocations: torch.Tensor, spinUpIndices: tuple[int, ...], spinDownIndices: tuple[int, ...]):
        #TODO?: Add check to ensure spinUpIndices and spinDownIndices really do partition (0,1,...,numElectrons) properly
        #... or compute spinDownIndices from spinUpIndices?
        #Update: The first NUp electrons are spin up. The rest are spin down. TODO: Replace the tuples with just the cutoff index NUp.
        numBatches = electronLocations.shape[0]
        numElectrons = electronLocations.shape[1]
        
        gRaw = self.rescaledElectronDistances(electronLocations)
        gUpUp     = self.siluActivation(self.linear0[0](gRaw[:, spinUpIndices, :, :][:, :, spinUpIndices, :]))
        gUpDown   = self.siluActivation(self.linear0[1](gRaw[:, spinUpIndices, :, :][:, :, spinDownIndices, :]))
        gDownUp   = self.siluActivation(self.linear0[1](gRaw[:, spinDownIndices, :, :][:, :, spinUpIndices, :]))
        gDownDown = self.siluActivation(self.linear0[0](gRaw[:, spinDownIndices, :, :][:, :, spinDownIndices, :]))
        gTotal = torch.cat((torch.cat((gUpUp, gUpDown), dim=-2), torch.cat((gDownDown, gDownUp), dim=-2)), dim=-3)

        pairDistances = self.getDistanceMatrixFromLocations(electronLocations)
        filterUpUp     = self.radialFilterFunction(pairDistances[:, spinUpIndices, :][:, :, spinUpIndices], 0)
        filterUpDown   = self.radialFilterFunction(pairDistances[:, spinUpIndices, :][:, :, spinDownIndices], 1)
        filterDownUp   = self.radialFilterFunction(pairDistances[:, spinDownIndices, :][:, :, spinUpIndices], 1)
        filterDownDown = self.radialFilterFunction(pairDistances[:, spinDownIndices, :][:, :, spinDownIndices], 0)
        filterTotal = torch.cat((torch.cat((filterUpUp, filterUpDown), dim=-2), torch.cat((filterDownDown, filterDownUp), dim=-2)), dim=-3)
        #TODO: Check that gTotal and filterTotal now have shape [NUMBER_OF_BATCHES, NUMBER_OF_ELECTRONS, NUMBER_OF_ELECTRONS, EMBEDDING_SIZE]

        electronEmbeddings = torch.sum(filterTotal * gTotal, dim=-2)
        electronEmbeddings = self.siluAcitvation(self.linear1(electronEmbeddings))
        electronEmbeddings = self.siluAcitvation(self.linear2(electronEmbeddings))
        electronEmbeddings = self.siluAcitvation(self.linear3(electronEmbeddings))

        return electronEmbeddings

    def getDistanceMatrixFromLocations(self, electronLocations: torch.Tensor):
        numBatches = electronLocations.shape[0]
        numElectrons = electronLocations.shape[1]
        expandedElectronLocations = electronLocations.unsqueeze(2).expand(numBatches, numElectrons, numElectrons, 3).clone()
        distanceMatrix = expandedElectronLocations - torch.transpose(expandedElectronLocations, dim0=1, dim1=2)
        return distanceMatrix

    def rescaledElectronDistances(self, electronLocations: torch.Tensor):
        distanceMatrix = self.getDistanceMatrixFromLocations(electronLocations)
        distancesNormMatrix = torch.linalg.norm(distanceMatrix, ord=2, dim=3)
        unscaledConcatMatrix = torch.cat((distanceMatrix, distancesNormMatrix.unsqueeze(-1)), -1)
        scalings = torch.log(distanceMatrix + 1) / distancesNormMatrix.unsqueeze(-1)
        return scalings * unscaledConcatMatrix

    def radialFilterFunction(self, x: torch.Tensor, spinInvKronecker: int):
        result = torch.abs(x)
        result0 = torch.exp(-1. * torch.pow(self.filterScaling[spinInvKronecker](result), 2.))
        result0 = self.filterLinear00[spinInvKronecker](result0)
        result1 = self.filterLinear10[spinInvKronecker](x)
        result1 = self.filterLinear11[spinInvKronecker](result1)
        return self.filterLinear2[spinInvKronecker](result0 * result1)