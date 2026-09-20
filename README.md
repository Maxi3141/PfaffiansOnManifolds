# Pfaffians on Manifolds

Be it for mathematical theory, quasi-particles in graphene, or the fractional quantum Hall effect, the analysis of two-dimensional electrons is of great interest for mathematics and physics. This project uses the "Neural Pfaffian" architecture developed by Gao & Günnemann in "Neural Pfaffians: Solving Many Many-Electron Schrödinger Equations" (Neural Information Processing Systems (NeurIPS), 2024) and combines it with the theory presented in "Quantum mechanics of a constrained particle" (R.C.T. da Costa, 1981 in: Physical Review A, Volume 23, Number 4) to provide a modern and flexible approach to compute the lowest energy state of $N_e$ electrons constrained to a 2-sphere embedded in 3D space.

## Installation, Training, and Evaluation

For installation, clone the repository and install `numpy` and `pytorch` in its environment. Furthermore, some scripts for visualization require `matplotlib`. Everything was developed and tested under Windows using `numpy==2.4.6` and `torch==2.12.0+cu132` but MacOS and Linux and the newest versions of `torch` and `numpy` should work aswell.

To train a new model, adjust the parameters in `train.py` and then run the standard `python train.py` from the command line.   
In order to evaluate an already existing model, use the `evaluate.py` script. It additionally contains the "mode" parameter to either evaluate the computed wave function at given locations, to sample from the wave function, to compute the modes of the wave function, or to benchmark the model for debugging.  
Call `playground/Tests.py` to run unit tests.

The results that can be obtained this way are shown below.

## Theory

Let $\Gamma\subset\mathbb R^3$ be a two-dimensional surface embedded in three-dimensional Euclidean space. For this project $\Gamma=S^2$. Using the notation from (Deal II: Tutorial 38), let $x_S:\mathbb R^2\supset\hat S\to S\subset\Gamma$ be a local parametrization, let $Dx_S\in{\mathbb R}^{3\times 2}$ be its Jacobian and define the first fundamental form $G_S:=(g_{ij})_{ij}:=(Dx_S)^TDx_S\in\mathbb R^{2\times 2}$. For a sufficiently smooth function $u:\Gamma\to\mathbb R$ the surface gradient $\nabla _{\Gamma}$ can now be defined by

$$(\nabla_\Gamma u)\circ x_\Gamma:=Dx_\Gamma G_\Gamma^{-1}\nabla(u\circ x_\Gamma)$$

where $x_\Gamma$ refers to the corresponding local parameterizations $x_S$. If $\tilde u$ is a smooth extension of $u$ to an open neighborhood of $\Gamma$, the surface gradient can also be defined by

$$\nabla_\Gamma u = \nabla\tilde u-n(n\cdot\nabla\tilde u)=P\nabla\tilde u$$

where $n$ is the normal vector of $\Gamma$ and $P$ is the projection matrix onto the tangential space $T_x\Gamma$.  
The surface Laplace operator or Laplace-Beltrami operator $\nabla_\Gamma^2:=\nabla_\Gamma\cdot\nabla_\Gamma$ is now usually introduced as

$$\nabla_\Gamma^2 u:=\frac1{\sqrt{\det G_\Gamma}}\partial_i(\sqrt{\det G_\Gamma}g^{ij}\partial_j u)$$

using the Einstein sum notation. Here however, the embedded version is more helpful:

$$\nabla_\Gamma^2 u = \nabla^2\tilde u - n^TD^2\tilde un-(n\cdot\nabla\tilde u)(\nabla\cdot n-n^TDnn)$$

where $$\kappa:=(\nabla\cdot n-n^TDnn)$$ simplifies to $2/r^2$ on a sphere with radius $r$. This formulation can easily be evaluated by treating the neural Pfaffian as a function $\mathbb R^3\to\mathbb R$ that just happens to be only evaluated on the surface. The differential expressions are computed using `torch.autograd`.  
Following da Costa's work, the Surface Schrödinger equation becomes

$$i\hbar \partial_t\phi=(-\frac{\hbar^2}{2m}\nabla _\Gamma^2 + V_\Gamma)\phi$$

where $V_\Gamma$ is the surface potential $V_\Gamma=M^2-K$ for mean curvature $M$ and Gaussian curvature $K$. On the sphere one can easily see that $V_{S^2}=0$.  
If the computation of the electrostatic forces is now adjusted to only consider the tangential term, one obtains the complete Hamiltonian of a system of $N_e$ electrons analogously to the original neural Pfaffian version:

$$H=-\frac12\sum_{i=1}^{N_e}\nabla _\Gamma^2+\sum_{i<j}^{N_e}\frac{\Vert P_i(r_i-r_j)\Vert}{\Vert r_i-r_j\Vert^2} + \sum_{i=1}^{N_e}V_\Gamma(r_i)$$

## Implementation

The implementation follows the overall structure of the original neural Pfaffian implementation closely, but simplifies and shrinks the learned systems whenever possible to cut numerical cost and to allow this project to run on ordinary consumer hardware.  
Expressions of type $\nabla _\Gamma^2 u / u$ are evaluated using logarithmic differentiation with additional consideration of the curvature terms:
$$\frac{\nabla _\Gamma^2 u}{u}=\nabla^2\log\vert\tilde u\vert + \Vert\nabla\vert\tilde u\vert\Vert^2-n^T(D^2\log\vert\tilde u\vert+(\nabla\log\vert\tilde u\tilde)(\nabla\log\vert\tilde u\tilde)^T)n-\kappa(n\cdot\nabla\log\vert\tilde u\vert).$$  
Since the definition of the surface differential operators relies on the fact the the restriction $\tilde u\vert _\Gamma=u$ is smooth, one must always ensure that all used activation functions are differentiable to a high enough degree.  
The Pfaffian network is a composition of continuous functions and thus itself continuous. Furthermore, unlike the Euclidean space $\mathbb R^3$, the sphere is compact, so that $\phi$ takes on a finite maximum somewhere on the domain and the integral $\int\phi^2 dx$ is always naturally bounded. Thus, no envelope function is needed in this setting.  
In the original version of Neural Pfaffians the geometry of the space is given by embeddings of the molecule structure. Here, there is no molecule structure. The geometry is only accessible implicitly by assuming that one can obtain electron positions by sampling from a probability distribution on the manifold. `GeometryProvider.py` takes in this positional data and provides embeddings that can be used instead of the MetaGNN embeddings.

WARNING: To fully take advantage of the stabilizing effects of the Spring optimizer, one needs to have a large enough batch size to reliably compute the "dead" gradient direction that only increases the wave function's amplitude. However, large batch sizes are only feasible when GPUs can be used instead of CPU only. Since CUDA is not yet fully supported here, one has to find the right balance between speed and stability for now.

## Results

Measuring the quality of the results is not trivial since physical analogues to compare to are not directly available. Instead, one can consider the Thomson problem which asks for the optimal distribution of electrons thought of as point charges on the sphere. For this solutions are available as long as the number of electrons stays low. If one now trains a neural Pfaffian on the sphere and then computes the modes / local maxima of the distribution, one can compare the Thomson energy of these maxima to the optimal Thomson energy and to the Thomson energies one would expect for points that are just uniformly distributed on the sphere.

For two electrons (1 Up, 1 Down, 3 Orbitals) we get the following energy decay during the training process:
<img width="640" height="480" alt="AvgEnergy_Moving10Average_-1_10_Clip" src="https://github.com/user-attachments/assets/7fce5cb5-070d-463c-b99f-497b9822409b" />

The average Thomson energy of the local maxima starts close to the energy one would expect for uniformly random distributed points (Red line, 1.0) but moves towards the theoretically best possible energy (Green line, 0.5):
<img width="640" height="480" alt="ModeThomEng_Moving3Avg" src="https://github.com/user-attachments/assets/8c122b72-04f1-4668-a0b0-627c83c4ca7a" />

Furthermore, for the two electron case we can fix one electron (here at `(0,-1, 0)`) and plot the marginal distribution of the other electron which clearly pools around the opposite pole at `(0,1,0)`:
<img width="666" height="608" alt="MarginalDistribution" src="https://github.com/user-attachments/assets/fbe403ae-fdad-4ba6-95c3-1553d84c91df" />

For the more interesting four-electron case (2 Up, 2 Down, 6 Orbitals) we get the decay of the physical energy... 
<img width="640" height="480" alt="AvgEnergy_Moving10Average" src="https://github.com/user-attachments/assets/2ad1af6a-66f5-45be-9a3b-8e0bb771ed64" />

 ...and the Thomson energy as shown below:
<img width="640" height="480" alt="ModesThomsonEnergies_Moving3Average" src="https://github.com/user-attachments/assets/9a6442ec-8e42-4812-9a0e-00da80348102" />

## Future
Future work should include obvious improvements like the support for odd numbers of electrons, full CUDA support, or a more stable implementation of the Spring optimizer. In the long run, two options appear interesting:

1. Support for more general shapes: The sphere is to symmetric for the surface potential $V_\Gamma$ to be interesting. Also, more complex manifolds which can only be parametrized piece-wise could make use of the reusability of orbitals intrinsic to neural Pfaffians: Instead of the orbitals being reused on a per nucleus basis one could reuse the orbitals for every local parametrized patch of the manifold.

2. Support for more complex spin interactions: As mentioned in the introduction, 2D electrons play an important role in condensed matter physics and for example explain the fractional quantum Hall effect. These all, however, require the consideration of the interaction of the electrons with an external magnetic field. The Haldane sphere for example, which is an important tool in these disciplines, is remarkably similar to the structure analyzed in this project and could be an interesting target for future analysis. 
