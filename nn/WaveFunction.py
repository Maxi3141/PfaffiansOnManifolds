import torch
from nn import ElectronEmbedding
from computation import LinAlgBasics as laBasics

class MultiElectronWaveFunction(torch.nn.Module):
    def __init__(self, numElectrons, numOrbitals, numSpinUpElectrons, sphereRadius, particleMass, embeddingDim):
        super().__init__()
        self.embeddingDim = embeddingDim
        self.numPfaffians = 4
        self.numElectrons = numElectrons
        self.numOrbitals = numOrbitals
        self.numSpinUpElectrons = numSpinUpElectrons
        self.sphereRadius = sphereRadius
        self.particleMass = particleMass

        self.embeddingNetwork = ElectronEmbedding.ElectronEmbeddingNetwork(self.embeddingDim)
        self.pfaffianNetworks = torch.nn.ModuleList([SinglePfaffianNetwork(numElectrons, numOrbitals, self.embeddingDim, numSpinUpElectrons) for _ in range(self.numPfaffians)])
        self.pfaffianScalings = torch.nn.Linear(self.numPfaffians, 1, bias=False)

        self.jastrowLinear0 = torch.nn.Linear(self.embeddingDim, 16)
        self.jastrowLinear1 = torch.nn.Linear(16, 8)
        self.jastrowLinear2 = torch.nn.Linear(8, 1)
        self.jastrowActivation = torch.nn.SiLU()
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
        pfaffians = torch.stack([pfNetwork(electronEmbeddings) for pfNetwork in self.pfaffianNetworks], dim=1)
        result = torch.exp(jastrowScalings) * self.pfaffianScalings(pfaffians)
        return result

    def forwardJastrowScalings(self, electronEmbeddings, x):
        electronMLPs = []
        for i in range(self.numElectrons):
            mlpOut = self.jastrowActivation(self.jastrowLinear0(electronEmbeddings[:,i,:]))
            mlpOut = self.jastrowActivation(self.jastrowLinear1(mlpOut))
            mlpOut = self.jastrowActivation(self.jastrowLinear2(mlpOut))
            electronMLPs.append(mlpOut)
        mlpTerm = torch.sum(torch.stack(electronMLPs, dim=1), dim=1)

        electronPairDistances = torch.linalg.norm(self.embeddingNetwork.getDistanceMatrixFromLocations(x), ord=2, dim=-1)
        upUpPairs     = electronPairDistances[:, :self.numSpinUpElectrons, :][:, :, :self.numSpinUpElectrons]
        upDownPairs   = electronPairDistances[:, :self.numSpinUpElectrons, :][:, :, self.numSpinUpElectrons:]
        downUpPairs   = electronPairDistances[:, self.numSpinUpElectrons:, :][:, :, :self.numSpinUpElectrons]
        downDownPairs = electronPairDistances[:, self.numSpinUpElectrons:, :][:, :, self.numSpinUpElectrons:]
        unscaledSameTerm = (torch.sum(upUpPairs + self.jastrowAlphaSame, dim=(1,2)) + torch.sum(downDownPairs + self.jastrowAlphaSame, dim=(1,2))).unsqueeze(1)
        unscaledDiffTerm = (torch.sum(upDownPairs + self.jastrowAlphaDiff, dim=(1,2)) + torch.sum(downUpPairs + self.jastrowAlphaDiff, dim=(1,2))).unsqueeze(1)
        scaledSameTerm = -0.25 * self.jastrowBetaSame * self.jastrowAlphaSame**2 * unscaledSameTerm
        scaledDiffTerm = -0.5 * self.jastrowBetaDiff * self.jastrowAlphaDiff**2 * unscaledDiffTerm
        
        return mlpTerm + scaledSameTerm + scaledDiffTerm

    def getLogGradient(self, x):
        #TODO: Replace the list below with some more elegant tensor expression.
        gradientsBatch = []
        for batchIndex in range(x.shape[0]):
            self.zero_grad()
            inputTensor = x[batchIndex,:,:].unsqueeze(0)
            currentGradient = torch.autograd.grad(torch.log(torch.abs(self.forward(inputTensor))), self.parameters())
            gradientsBatch.append(currentGradient)
        return gradientsBatch
        
    def updateWeights(self, gradient, learningRate):
        with torch.no_grad():
            for currentParams, newGradient in zip(self.parameters(), gradient):
                newGradient = newGradient.clamp(-1e-3, 1e-3)
                currentParams += -learningRate * newGradient

class SinglePfaffianNetwork(torch.nn.Module):
    def __init__(self, numElectrons, numOrbitals, embeddingSize, numUpElectrons):
        super().__init__()
        self.numElectrons = numElectrons
        self.numOrbitals = numOrbitals
        self.embeddingSize = embeddingSize
        self.numUpElectrons = numUpElectrons

        self.readoutLayersA          = torch.nn.ModuleList([torch.nn.Linear(2 * self.embeddingSize, 1, bias=False) for _ in range(self.numOrbitals**2)])
        self.readoutLayersDiagPhi    = torch.nn.ModuleList([torch.nn.Linear(self.embeddingSize, 1, bias=False) for _ in range(self.numOrbitals)])
        self.readoutLayersOffdiagPhi = torch.nn.ModuleList([torch.nn.Linear(self.embeddingSize, 1, bias=False) for _ in range(self.numOrbitals)])

        for layer in self.readoutLayersA:
            torch.nn.init.normal_(layer.weight, mean=0., std=1.)
        for layer in self.readoutLayersDiagPhi:
            torch.nn.init.normal_(layer.weight, mean=0., std=1.)
        for layer in self.readoutLayersOffdiagPhi:
            torch.nn.init.normal_(layer.weight, mean=0., std=1.)

    def forward(self, electronEmbeddings):
        A = self.forwardA(electronEmbeddings)
        phi = self.forwardDiagonalPhi(electronEmbeddings)
        phiHat = self.forwardOffdiagonalPhi(electronEmbeddings)

        phiUp      = phi[:,:self.numUpElectrons,:]
        phiDown    = phi[:,self.numUpElectrons:,:]
        phiHatUp   = phiHat[:,:self.numUpElectrons,:]
        phiHatDown = phiHat[:,self.numUpElectrons:,:]
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
        readoutsA = torch.stack(rawReadoutsList, dim=1).squeeze().reshape([numBatches, self.numOrbitals, self.numOrbitals])
        aDiag = 0.5 * (readoutsA - torch.transpose(readoutsA, dim0=1, dim1=2))
        aOffDiag = 0.5 * (readoutsA + torch.transpose(readoutsA, dim0=1, dim1=2))
        aComplete = torch.cat((torch.cat((aDiag, aOffDiag), dim=2), torch.cat((-aOffDiag, aDiag), dim=2)), dim=1)
        return aComplete

    def forwardDiagonalPhi(self, electronEmbeddings):
        # electronEmbedding.shape = [NUM_BATCHES, NUM_ELECTRONS, EMBEDDING_DIM]
        # self.readoutLayersDiagPhi[i](electronEmbeddings).shape = [NUM_BATCHES, NUM_ELECTRONS, 1]
        readoutsPhiList = []
        for readoutLayer in self.readoutLayersDiagPhi:
            currentReadOut = readoutLayer(electronEmbeddings)
            readoutsPhiList.append(currentReadOut)
        readoutsPhi = torch.cat(readoutsPhiList, dim=2)
       
        # readoutsPhi.shape = [NUM_BATCHES, NUM_ELECTRONS, NUM_ORBITALS]
        return readoutsPhi

    def forwardOffdiagonalPhi(self, electronEmbeddings):
        readoutsOffdiagPhi = torch.cat([self.readoutLayersOffdiagPhi[i](electronEmbeddings) for i in range(self.numOrbitals)], dim=2)
        return readoutsOffdiagPhi