import torch

class MultiElectronWaveFunction(torch.nn.Module):
    def __init__(self, numElectrons, numOrbitals, particleMass, sphereRadius):
        self.numElectrons = numElectrons
        self.numOrbitals = numOrbitals
        self.particleMass = particleMass
        self.sphereRadius = sphereRadius

        self.readoutsA = torch.nn.ModuleList([torch.nn.Linear(numElectrons**2, 1, bias=False) for _ in range(self.numElectrons**2)])

#TODO: Check whether two seperate Moon embeddings are necessary or if the Diags and Offdiags both get fed the same embeddings.

    def forward(self, x):
        A = self.forwardA(self, x)
        Phi = self.forwardDiagonalPhi(self, x)
        PhiHat = self.forwardOffdiagonalPhi(self, x)
        #Hier koennte ihre Werbung stehen.

    def forwardA(self, electronEmbeddings):
        numBatches = electronEmbeddings.shape[0]
        numElectrons = electronEmbeddings.shape[1]
        embeddingSize = electronEmbeddings.shape[2]
        expandedEmbeddings = electronEmbeddings.unsqueeze(2).expand(numBatches, numElectrons, numElectrons, embeddingSize).clone()
        pairEmbeddings = torch.cat((expandedEmbeddings, torch.transpose(expandedEmbeddings, 1, 2)), dim=-1)
        #"pairEmbeddings" now has shape [NUMBER_OF_BATCHES, NUMBER_OF_ELECTRONS, NUMBER_OF_ELECTRONS, 2 * EMBEDDING_DIM].
        usablePairEmbeddings = torch.flatten(pairEmbeddings, start_dim=1, end_dim=2)
        

        #TODO: Add linear read outs.
        #TODO: Assemble A from antisymmetric and symmtric part.
        pass

    def forwardDiagonalPhi(self, x):
        pass

    def forwardOffdiagonalPhi(self, x):
        pass

    def getElectronEmbedding(self, electronLocations):
        pass

    def getLogGradient(self, x):
        return torch.autograd.grad(torch.log(torch.abs(self.forward(x))), self.parameters())
    
    def updateWeights(self, gradient, learningRate):
        with torch.no_grad():
            for currentParams, newGradient in zip(self.parameters(), gradient):
                newGradient = newGradient.clamp(-1e-3, 1e-3)
                currentParams += -learningRate * newGradient