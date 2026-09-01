import torch 
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from decimal import Decimal

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from nn import WaveFunction

nPhi = 100
nTheta = 100

numElectrons       = 2
numSpinUpElectrons = 1
numOrbitals        = 3
sphereRadius       = 1.0
particleMass       = 1e-1

embeddingDim    = 128
numOrbitalParams = 32
useCuda         = False

phi = np.linspace(0, np.pi, nPhi)
theta = np.linspace(0, 2 * np.pi, nTheta)

x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

torchInputLocations = torch.cat([
    torch.stack([
        torch.tensor([[-1., 0., 0.], [np.sin(p) * np.cos(t), np.sin(p) * np.sin(t), np.cos(p)]], dtype=torch.float)
    for p in phi]) 
for t in theta])

waveNetwork = WaveFunction.MultiElectronWaveFunction(numElectrons, numOrbitals, numSpinUpElectrons, sphereRadius, particleMass, embeddingDim, numOrbitalParams)
print(f"Created wave function Pfaffians with {sum(p.numel() for p in waveNetwork.parameters())} parameters in total.")
strMass = '%.2E' % Decimal(particleMass)
waveNetwork.load_state_dict(torch.load(f"./saves/ManifoldPfaffian_{numElectrons}E_{numOrbitals}O_{numSpinUpElectrons}Up_M{strMass}.pth"))

print("Shape of input for network:", torchInputLocations.shape)
waveEval = waveNetwork(torchInputLocations).reshape([nPhi, nTheta])
probEval = torch.pow(waveEval, 2.)
print("... got result of shape", waveEval.shape)

phi, theta = np.meshgrid(phi, theta)

# The Cartesian coordinates of the unit sphere
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

# Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
fcolors = probEval.detach().numpy()
fmax, fmin = fcolors.max(), fcolors.min()
fcolors = (fcolors - fmin) / (fmax - fmin)

# Set the aspect ratio to 1 so our sphere looks spherical
fig = plt.figure(figsize=plt.figaspect(1.0))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.seismic(fcolors))
# Turn off the axis planes
#ax.set_axis_off()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()