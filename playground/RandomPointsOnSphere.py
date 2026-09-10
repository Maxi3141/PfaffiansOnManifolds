import numpy as np

n_batches = 128
n_points = 4

energies = []

for i in range(n_batches):
    point_positions = []
    for j in range(n_points):
        theta = 2.0 * np.pi * np.random.uniform(0.0, 1.0)
        phi = np.arccos(2.0 * np.random.uniform(0.0, 1.0) - 1.0)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        point_positions.append((x, y, z))

    total_energy = 0.0
    for pointI in range(n_points-1):
        for pointJ in range(pointI+1, n_points):
            locI = point_positions[pointI]
            locJ = point_positions[pointJ]
            r = ((locI[0] - locJ[0])**2 + (locI[1] - locJ[1])**2 + (locI[2] - locJ[2])**2)**0.5
            total_energy += 1 / r
    energies.append(total_energy)

print("Finished", n_batches, "batches of", n_points, "points. Min:", str(min(energies)), ", Max:", str(max(energies)), ", Avg:", str(sum(energies) / len(energies)))
