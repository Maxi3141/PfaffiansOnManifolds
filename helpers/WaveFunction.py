import torch

class MultiElectronWaveFunction(torch.nn.Module):
    def __init__(self, numElectrons, particleMass, sphereRadius):
        self.numElectrons = numElectrons
        self.particleMass = particleMass
        self.sphereRadius = sphereRadius

    def forward(self, x):
        pass
        #Hier koennte ihre Werbung stehen.

    def getLogGradient(self, x):
        return torch.autograd.grad(torch.log(self.forward(x)), self.parameters())