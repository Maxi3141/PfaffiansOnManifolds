import numpy as np
import StatBasics

def estimateVariationalEnergy(waveFunction, numElectrons, sphereRadius):
    numSamples = 10
    electronSamples = StatBasics.sampleFromWaveFunction(waveFunction, numElectrons, sphereRadius)
    