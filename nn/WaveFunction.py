import torch
from nn import ElectronEmbedding
from nn import GeometryProvider
from computation import LinAlgBasics as laBasics

class MultiElectronWaveFunction(torch.nn.Module):
    def __init__(self, numElectrons, numOrbitals, numSpinUpElectrons, sphereRadius, particleMass, embeddingDim, numOrbitalParams, numPfaffians):
        super().__init__()
        self.embeddingDim = embeddingDim
        self.numOrbitalParams = numOrbitalParams
        self.numPfaffians = numPfaffians
        self.numElectrons = numElectrons
        self.numOrbitals = numOrbitals
        self.numSpinUpElectrons = numSpinUpElectrons
        self.sphereRadius = sphereRadius
        self.particleMass = particleMass

        self.orbitalParametrizer = GeometryProvider.GeometryProvider(self.numOrbitalParams)
        self.embeddingNetwork = ElectronEmbedding.ElectronEmbeddingNetwork(self.embeddingDim)
        self.pfaffiansNetwork = MultiPfaffianNetwork(numElectrons, numOrbitals, self.embeddingDim, self.numOrbitalParams, numSpinUpElectrons, self.numPfaffians)
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
        orbitalParams = self.orbitalParametrizer(x)

        jastrowScalings = self.forwardJastrowScalings(electronEmbeddings, x)
        pfaffians = torch.transpose(self.pfaffiansNetwork(electronEmbeddings, orbitalParams), dim0=0, dim1=1)
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
        unscaledSameTerm = (torch.sum(torch.clamp(1. / (upUpPairs + self.jastrowAlphaSame), -1e+3, 1e+3), dim=(1,2)) + torch.sum(torch.clamp(1. / (downDownPairs + self.jastrowAlphaSame), -1e+3, 1e+3), dim=(1,2))).unsqueeze(1)
        unscaledDiffTerm = (torch.sum(torch.clamp(1. / (upDownPairs + self.jastrowAlphaDiff), -1e+3, 1e+3), dim=(1,2)) + torch.sum(torch.clamp(1. / (downUpPairs + self.jastrowAlphaDiff), -1e+3, 1e+3), dim=(1,2))).unsqueeze(1)
        scaledSameTerm = -0.25 * self.jastrowBetaSame * self.jastrowAlphaSame**2 * unscaledSameTerm
        scaledDiffTerm = -0.5 * self.jastrowBetaDiff * self.jastrowAlphaDiff**2 * unscaledDiffTerm
        
        return mlpTerm + scaledSameTerm + scaledDiffTerm

    def getLogGradient(self, x, logStabilizer: float = 1e-7):
        subParameterGradients = [torch.zeros([x.shape[0]] + list(p.shape)) for p in self.parameters()]
        for batchIndex in range(x.shape[0]):
            self.zero_grad()
            inputTensor = x[batchIndex,:,:].unsqueeze(0)
            currentGradient = torch.autograd.grad(torch.log(torch.abs(self.forward(inputTensor)) + logStabilizer), self.parameters())
            for subParameterIndex, subGradient in enumerate(currentGradient):
                subParameterGradients[subParameterIndex][batchIndex,...] = subGradient
        return subParameterGradients
        
    def updateWeights(self, gradient, learningRate):
        with torch.no_grad():
            for currentParams, newGradient in zip(self.parameters(), gradient):
                newGradient = newGradient.clamp(-1e-3, 1e-3)
                currentParams += -learningRate * newGradient

class MultiPfaffianNetwork(torch.nn.Module):
    def __init__(self, numElectrons, numOrbitals, embeddingSize, numOrbitalParams, numUpElectrons, numPfaffians):
        super().__init__()
        self.numElectrons     = numElectrons
        self.numOrbitals      = numOrbitals
        self.embeddingSize    = embeddingSize
        self.numOrbitalParams = numOrbitalParams
        self.numUpElectrons   = numUpElectrons
        self.numPfaffians     = numPfaffians

        self.readoutAWeights          = torch.nn.Parameter(torch.empty(self.numPfaffians, self.numOrbitals * self.numOrbitals, self.numOrbitalParams))
        self.readoutDiagPhiWeights    = torch.nn.Parameter(torch.empty(self.numPfaffians, self.embeddingSize * self.numOrbitals, self.numOrbitalParams))
        self.readoutOffDiagPhiWeights = torch.nn.Parameter(torch.empty(self.numPfaffians, self.embeddingSize * self.numOrbitals, self.numOrbitalParams))

        torch.nn.init.normal_(self.readoutAWeights, mean=0., std=1.)
        torch.nn.init.normal_(self.readoutDiagPhiWeights, mean=0., std=1.)
        torch.nn.init.normal_(self.readoutOffDiagPhiWeights, mean=0., std=.01)

    def forward(self, electronEmbeddings, orbitalParams):
        A = self.forwardA(electronEmbeddings, orbitalParams)
        phi = self.forwardDiagonalPhi(electronEmbeddings, orbitalParams)
        phiHat = self.forwardOffdiagonalPhi(electronEmbeddings, orbitalParams)

        phiUp      = phi[:,:,:self.numUpElectrons,:]
        phiDown    = phi[:,:,self.numUpElectrons:,:]
        phiHatUp   = phiHat[:,:,:self.numUpElectrons,:]
        phiHatDown = phiHat[:,:,self.numUpElectrons:,:]
        orbitElecPairing = torch.cat((torch.cat((phiUp, phiHatUp), dim=-1), torch.cat((phiHatDown, phiDown), dim=-1)), dim=-2)
        
        orbitAOrbit = torch.matmul(torch.matmul(orbitElecPairing, A), torch.transpose(orbitElecPairing, dim0=-2, dim1=-1))

        pfOrbitAOrbit = laBasics.getPfaffian(orbitAOrbit)
        pfA           = laBasics.getPfaffian(A)

        return pfOrbitAOrbit / pfA

    #TODO: The entire structure of this will have to be reworked once more complicated manifolds with multiple charts are supported.
    def forwardA(self, electronEmbeddings, orbitalParams):
        batchSize = electronEmbeddings.shape[0]

        readoutsA = torch.einsum("poi,bi->pbo", self.readoutAWeights, orbitalParams).reshape([self.numPfaffians, batchSize, self.numOrbitals, self.numOrbitals])

        aDiag = 0.5 * (readoutsA - torch.transpose(readoutsA, dim0=-1, dim1=-2))
        aOffDiag = 0.5 * (readoutsA + torch.transpose(readoutsA, dim0=-1, dim1=-2))
        aComplete = torch.cat((torch.cat((aDiag, aOffDiag), dim=-1), torch.cat((-aOffDiag, aDiag), dim=-1)), dim=-2)
        return aComplete

    def forwardDiagonalPhi(self, electronEmbeddings, orbitalParams):
        batchSize = electronEmbeddings.shape[0]
        
        projectionW = torch.einsum("poi,bi->pbo", self.readoutDiagPhiWeights, orbitalParams).reshape([self.numPfaffians, batchSize, self.embeddingSize, self.numOrbitals])
        result = torch.einsum("bnd,pbdo->pbno", electronEmbeddings, projectionW)
        return result

    def forwardOffdiagonalPhi(self, electronEmbeddings, orbitalParams):
        batchSize = electronEmbeddings.shape[0]

        projectionWHat = torch.einsum("poi,bi->pbo", self.readoutOffDiagPhiWeights, orbitalParams).reshape([self.numPfaffians, batchSize, self.embeddingSize, self.numOrbitals])
        result = torch.einsum("bnd,pbdo->pbno", electronEmbeddings, projectionWHat)
        return result