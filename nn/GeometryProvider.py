import torch

# In the original paper by Gao & Guennemann (2024) the external geometry is determined
# by the locations and the charges of the nuclei and is extracted using MetaGNN.
# We do not have nuclei or any equivalent structure. Here, the geometry is only accesible
# indirectly via the sampled electrons since they are always located on the manifold.
# "GeometryProvider" takes in the locations of the sampled electrons and returns 
# parameters which can be used instead of the output of MetaGNN.

class GeometryProvider(torch.nn.Module):
    def __init__(self, geometryEmbeddingDim):
        super().__init__()
        self.geoEmbeddingDim = geometryEmbeddingDim

        self.embActivation = torch.nn.functional.silu

        self.embeddingMatrix = torch.nn.Linear(3, self.geoEmbeddingDim, bias=False)
        self.linear0 = torch.nn.Linear(self.geoEmbeddingDim, self.geoEmbeddingDim)
        self.linear1 = torch.nn.Linear(self.geoEmbeddingDim, self.geoEmbeddingDim)

        self.linear2 = torch.nn.Linear(self.geoEmbeddingDim, self.geoEmbeddingDim)
        self.linear3 = torch.nn.Linear(self.geoEmbeddingDim, self.geoEmbeddingDim)

    #"electronLocations" has dimensions [NUMBER_OF_BATCHES, NUMBER_OF_ELECTRONS, 3]
    def forward(self, electronLocations: torch.Tensor):
        electronDistances = self.getDistanceMatrixFromLocations(electronLocations)
        singleEmbeddings = torch.sum(self.embActivation(self.embeddingMatrix(electronDistances)), dim=-2)
        singleEmbeddings = self.embActivation(self.linear0(singleEmbeddings))
        singleEmbeddings = self.embActivation(self.linear1(singleEmbeddings))

        cumulEmbeddings = torch.sum(singleEmbeddings, dim=-2)
        cumulEmbeddings = self.embActivation(self.linear2(cumulEmbeddings))
        cumulEmbeddings = self.embActivation(self.linear3(cumulEmbeddings))

        return cumulEmbeddings

    def getDistanceMatrixFromLocations(self, electronLocations: torch.Tensor):
        batchSize = electronLocations.shape[0]
        numElectrons = electronLocations.shape[1]
        expandedElectronLocations = electronLocations.unsqueeze(2).expand(batchSize, numElectrons, numElectrons, 3).clone()
        distanceMatrix = expandedElectronLocations - torch.transpose(expandedElectronLocations, dim0=1, dim1=2)
        return distanceMatrix