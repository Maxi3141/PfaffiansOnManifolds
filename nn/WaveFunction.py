import torch
from nn import ElectronEmbedding
from computation import LinAlgBasics as laBasics

class MultiElectronWaveFunction(torch.nn.Module):
    def __init__(self, numElectrons, numOrbitals, numSpinUpElectrons):
        self.embeddingDim = 256
        self.numPfaffians = 4
        self.numElectrons = numElectrons
        self.numOrbitals = numOrbitals
        self.numSpinUpElectrons = numSpinUpElectrons

        self.embeddingNetwork = ElectronEmbedding(self.embeddingDim)
        self.pfaffianNetworks = torch.nn.ModuleList([SinglePfaffianNetwork(numElectrons, numOrbitals, self.embeddingDim) for _ in range(self.numPfaffians)])
        self.pfaffianScalings = torch.nn.Linear(self.numPfaffians, 1, bias=False)

        self.jastrowLinear0 = torch.nn.Linear(self.embeddingDim, 16)
        self.jastrowLinear1 = torch.nn.Linear(16, 8)
        self.jastrowLinear2 = torch.nn.Linear(8, 1)
        self.jastrowActivation = torch.nn.SiLU
        self.jastrowAlphaSame = torch.nn.Parameter(torch.tensor(1.))
        self.jastrowAlphaDiff = torch.nn.Parameter(torch.tensor(1.))
        self.jastrowBetaSame = torch.nn.Parameter(torch.tensor(1.))
        self.jastrowBetaDiff = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        spinUpIndices = tuple(range(self.numSpinUpElectrons))
        spinDownIndices = tuple(range(self.numSpinUpElectrons, self.numElectrons))
        electronEmbeddings = self.embeddingNetwork(x, spinUpIndices, spinDownIndices)

        #TODO: Add vmap stuff to the "forwards" below
        jastrowScalings = self.forwardJastrowScalings(electronEmbeddings, x)
        pfaffians = torch.cat([pfNetwork(electronEmbeddings) for pfNetwork in self.pfaffianNetworks], dim=1)

        return torch.exp(jastrowScalings) * self.pfaffianScalings(pfaffians)

    def forwardJastrowScalings(self, electronEmbeddings, x):
        electronMLPs = []
        for i in range(self.numElectrons):
            mlpOut = self.jastrowActivation(self.jastrowLinear0(electronEmbeddings[:,i,:].squeeze()))
            mlpOut = self.jastrowActivation(self.jastrowLinear1(mlpOut))
            mlpOut = self.jastrowActivation(self.jastrowLinear2(mlpOut))
            electronMLPs.append(mlpOut)
        mlpTerm = torch.sum(torch.cat(electronMLPs, dim=1), dim=1)

        electronPairDistances = self.embeddingNetwork.getDistanceMatrixFormLocations(x)
        upUpPairs     = electronPairDistances[:, :self.numSpinUpElectrons, :][:, :, :self.numSpinUpElectrons]
        upDownPairs   = electronPairDistances[:, :self.numSpinUpElectrons, :][:, :, self.numSpinUpElectrons:]
        downUpPairs   = electronPairDistances[:, self.numSpinUpElectrons:, :][:, :, :self.numSpinUpElectrons]
        downDownPairs = electronPairDistances[:, self.numSpinUpElectrons:, :][:, :, self.numSpinUpElectrons:]
        unscaledSameTerm = torch.sum(upUpPairs + self.jastrowAlphaSame, dim=(1,2)) + torch.sum(downDownPairs + self.jastrowAlphaSame, dim=(1,2))
        unscaledDiffTerm = torch.sum(upDownPairs + self.jastrowAlphaDiff, dim=(1,2)) + torch.sum(downUpPairs + self.jastrowAlphaDiff, dim=(1,2))
        scaledSameTerm = -0.25 * self.jastrowBetaSame * self.jastrowAlphaSame**2 * unscaledSameTerm
        scaledDiffTerm = -0.5 * self.jastrowBetaDiff * self.jastrowAlphaDiff**2 * unscaledDiffTerm

        return mlpTerm + scaledSameTerm + scaledDiffTerm

    def getLogGradient(self, x):
        return torch.autograd.grad(torch.log(torch.abs(self.forward(x))), self.parameters())
        
    def updateWeights(self, gradient, learningRate):
        with torch.no_grad():
            for currentParams, newGradient in zip(self.parameters(), gradient):
                newGradient = newGradient.clamp(-1e-3, 1e-3)
                currentParams += -learningRate * newGradient

class SinglePfaffianNetwork(torch.nn.Module):
    def __init__(self, numElectrons, numOrbitals, embeddingSize, numUpElectrons):
        self.numElectrons = numElectrons
        self.numOrbitals = numOrbitals
        self.embeddingSize = embeddingSize
        self.numUpElectrons = numUpElectrons

        self.readoutLayersA = torch.nn.ModuleList([torch.nn.Linear(2 * self.embeddingSize, 1, bias=False) for _ in range(self.numElectrons**2)])
        self.readoutLayersDiagPhi = torch.nn.ModuleList([torch.nn.Linear(self.embeddingSize, 1, bias=False) for _ in range(self.numElectrons * self.numOrbitals)])
        self.readoutLayersOffdiagPhi = torch.nn.ModuleList([torch.nn.Linear(self.embeddingSize, 1, bias=False) for _ in range(self.numOrbitals)])

    def forward(self, electronEmbeddings):
        A = self.forwardA(self, electronEmbeddings)
        phi = self.forwardDiagonalPhi(self, electronEmbeddings)
        phiHat = self.forwardOffdiagonalPhi(self, electronEmbeddings)

        phiUp      = phi[:,:self.numUpElectrons,:]
        phiDown    = phi[:,self.numUpElectrons:self.numUpElectrons,:]
        phiHatUp   = phiHat[:,:self.numUpElectrons,:]
        phiHatDown = phiHat[:,self.numUpElectrons:self.numUpElectrons,:]
        orbitElecPairing = torch.cat((torch.cat((phiUp, phiHatUp), dim=2), torch.cat((phiHatDown, phiDown), dim=2)), dim=1)
        orbitAOrbit = torch.matmul(torch.matmul(orbitElecPairing, A), torch.transpose(orbitElecPairing, dim0=1, dim1=2))
        return torch.vmap(laBasics.getPfaffian)(orbitAOrbit)

    def forwardA(self, electronEmbeddings):
        numBatches = electronEmbeddings.shape[0]
        numElectrons = electronEmbeddings.shape[1]
        expandedEmbeddings = electronEmbeddings.unsqueeze(2).expand(numBatches, numElectrons, numElectrons, self.embeddingSize).clone()
        pairEmbeddings = torch.cat((expandedEmbeddings, torch.transpose(expandedEmbeddings, 1, 2)), dim=-1)
        #"pairEmbeddings" now has shape [NUMBER_OF_BATCHES, NUMBER_OF_ELECTRONS, NUMBER_OF_ELECTRONS, 2 * EMBEDDING_DIM].
        usablePairEmbeddings = torch.flatten(pairEmbeddings, start_dim=1, end_dim=2)
        rawReadoutsList = []
        for layerIndex, layer in enumerate(self.readoutLayersA):
            rawReadoutsList.append(layer(usablePairEmbeddings[:,layerIndex,:]))
        readoutsA = torch.stack(rawReadoutsList, dim=1).squeeze().reshape([numBatches, numElectrons, numElectrons])
        aDiag = 0.5 * (readoutsA - torch.transpose(readoutsA, dim0=1, dim1=2))
        aOffDiag = 0.5 * (readoutsA + torch.transpose(readoutsA, dim0=1, dim1=2))
        aComplete = torch.cat((torch.cat((aDiag, aOffDiag), dim=2), torch.cat((-aOffDiag, aDiag), dim=2)), dim=1)
        return aComplete

    def forwardDiagonalPhi(self, electronEmbeddings):
        # electronEmbedding.shape = [NUM_BATCHES, NUM_ELECTRONS, EMBEDDING_DIM]
        # self.readoutLayersDiagPhi[i](electronEmbeddings).shape = [NUM_BATCHES, NUM_ELECTRONS, 1]
        readoutsPhi = torch.cat([self.readoutLayersDiagPhi[i](electronEmbeddings) for i in range(self.numOrbitals)], dim=2)
        # readoutsPhi.shape = [NUM_BATCHES, NUM_ELECTRONS, NUM_ORBITALS]
        return readoutsPhi

    def forwardOffdiagonalPhi(self, electronEmbeddings):
        readoutsOffdiagPhi = torch.cat([self.readoutLayersDiagPhi[i](electronEmbeddings) for i in range(self.numOrbitals)], dim=2)
        return readoutsOffdiagPhi