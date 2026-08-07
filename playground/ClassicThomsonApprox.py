import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from computation import StatBasics as stats
import numpy as np

n_points = 200
point_forces = []
point_positions = []

num_timesteps = 2000

force_coeff = 1

#Generate initial positions
for i in range(n_points):
    point_positions.append(list(stats.sampleUniformOnSphere()))

for timeIndex in range(num_timesteps):

    #Compute forces
    point_forces = [[0, 0, 0] for _ in range(n_points)]
    for pointI in range(n_points-1):
        for pointJ in range(pointI+1, n_points):
            locI = point_positions[pointI]
            locJ = point_positions[pointJ]
            r = ((locI[0] - locJ[0])**2 + (locI[1] - locJ[1])**2 + (locI[2] - locJ[2])**2)**0.5
            for k in range(3):
                point_forces[pointI][k] += (locJ[k] - locI[k]) / r
                point_forces[pointJ][k] += (locI[k] - locJ[k]) / r

    #Apply forces
    for pointI in range(n_points):
        for k in range(3):
            point_positions[pointI][k] += force_coeff * point_forces[pointI][k]

    #Project points back onto sphere
    for pointI in range(n_points):
        locI = point_positions[pointI]
        point_norm = (locI[0]**2 + locI[1]**2 + locI[2]**2)**0.5
        for k in range(3):
            point_positions[pointI][k] = locI[k] / point_norm

    #Compute energy
    total_energy = 0.0
    for pointI in range(n_points-1):
        for pointJ in range(pointI+1, n_points):
            locI = point_positions[pointI]
            locJ = point_positions[pointJ]
            r = ((locI[0] - locJ[0])**2 + (locI[1] - locJ[1])**2 + (locI[2] - locJ[2])**2)**0.5
            total_energy += 1 / r
    
    print("Energy after", timeIndex, "time steps:", total_energy)

visual_points_x = []
visual_points_y = []
visual_points_z = []
for loc in point_positions:
    visual_points_x.append(loc[0])
    visual_points_y.append(loc[1])
    visual_points_z.append(loc[2])

#Set colours and render
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(visual_points_x,visual_points_y,visual_points_z,color="k",s=20)

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
ax.set_aspect("equal")
plt.tight_layout()
plt.show()
