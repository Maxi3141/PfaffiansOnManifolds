import torch
from collections.abc import Callable

# Functions to deal the manifolds that are diffeomorphic to the sphere S**2.
# "manifoldMap" is always assumed to be a diffeomorphism that takes a point on the sphere and maps it to the corresponding point on the manifold.

# All tensors that contain coordinates are expected to have the coords in the last dimensions.
# ...so the tensor shapes are [..., 2] in flat reference space or [..., 3] in embedding space.
# The flat reference element is always expected to be the unit square [0,1]**2
# The sphere's radius is always expeted to be 1. Other radii will have to be realized using the diffeomorphism.

# See Wikipedia (or any other source) for correspondence between fundamental forms and E, G, F, L, N, and M.

#TODO: Is the inversion of the diffeomorphism needed somewhere?
#TODO: Transformations expect input coordinates to be on the sphere. Would input coordinates in flat reference space be more suitable?
#      ...computation would probably be simpler but unit square as default reference space could feel unnatural.

def transformationName(sphereTransform: Callable[[torch.Tensor], torch.Tensor]):
    if sphereTransform == None:
        return "Sphere"
    if sphereTransform.__name__ == "ellipsoidTransform":
        return "Ellipsoid"
    if sphereTransform.__name__ == "eggTransform":
        return "Egg"
    return "Unknown"

def ellipsoidTransform(sphereCoords: torch.Tensor):
    ellipsoidCoords = torch.zeros_like(sphereCoords)
    ellipsoidCoords[...,0] = 2. * sphereCoords[...,0]
    ellipsoidCoords[...,1] = sphereCoords[...,1]
    ellipsoidCoords[...,2] = sphereCoords[...,2]
    return ellipsoidCoords

def eggTransform(sphereCoords: torch.Tensor):
    flatCoords = getFlatCoordsFromSphereCoords(sphereCoords)
    u = flatCoords[...,0]
    v = flatCoords[...,1]
    r0 = 0.6666 * (1. - 0.25 * torch.cos(torch.pi * v)) * torch.sin(torch.pi * v) * torch.cos(2. * torch.pi * u)
    r1 = 0.6666 * (1. - 0.25 * torch.cos(torch.pi * v)) * torch.sin(torch.pi * v) * torch.sin(2. * torch.pi * u)
    r2 = torch.cos(torch.pi * v)
    return torch.stack([r0, r1, r2], dim=-1).to(torch.float32)

def getSphereCoordsFromFlatCoords(flatCoords: torch.Tensor):
    theta = 2. * torch.pi * flatCoords[..., 0]
    phi = torch.pi * flatCoords[..., 1]
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    return torch.stack([x, y, z], dim=-1).to(torch.float32)

def getFlatCoordsFromSphereCoords(sphereCoords: torch.Tensor):
    x = sphereCoords[...,0]
    y = sphereCoords[...,1]
    z = sphereCoords[...,2]
    v = torch.acos(z)
    u = torch.atan2(y, x)
    v = v / torch.pi
    u = u / (2. * torch.pi)
    u = u - torch.floor(u)
    return torch.stack([u, v], dim=-1)

def getManifoldCoordsFromSphereCoords(sphereCoords: torch.Tensor, manifoldMap):
    return manifoldMap(sphereCoords)

def getManifoldCoordsFromFlatCoords(flatCoords: torch.Tensor, manifoldMap):
    return manifoldMap(getSphereCoordsFromFlatCoords(flatCoords))

def getSphereTangentBasisFromFlatCoords(flatCoords: torch.Tensor):
    u = 2. * torch.pi * flatCoords[...,0]
    v = torch.pi * flatCoords[...,1]
    tan0 = torch.stack([-torch.sin(v) * torch.sin(u), torch.sin(v) * torch.cos(u), torch.zeros_like(u)], dim=-1)
    tan1 = torch.stack([torch.cos(v) * torch.cos(u), torch.cos(v) * torch.sin(u), -torch.sin(v)], dim=-1)
    tanBasis = torch.stack([tan0, tan1], dim=-2)
    tanUnitBasis = torch.nn.functional.normalize(tanBasis, dim=-1)
    return tanUnitBasis.to(torch.float32)

def getSphereTangentBasisFromSphereCoords(sphereCoords: torch.Tensor):
    flatCoords = getFlatCoordsFromSphereCoords(sphereCoords)
    return getSphereTangentBasisFromFlatCoords(flatCoords)

def getManifoldTangentBasisFromSphereCoords(sphereCoords: torch.Tensor, manifoldMap):
    targetDiffeoJacobiShape = list(sphereCoords.shape[0:-1]) + [3, 3]
    singularCoords = torch.flatten(sphereCoords, start_dim=0, end_dim=-2)
    diffeoJacobis = torch.reshape(torch.vmap(torch.func.jacfwd(manifoldMap))(singularCoords), targetDiffeoJacobiShape)
    diffeoJacobis = torch.repeat_interleave(diffeoJacobis.unsqueeze(-3), repeats=2, dim=-3).to(torch.float32)
    sphereTangentialBasis = getSphereTangentBasisFromSphereCoords(sphereCoords).unsqueeze(-1)
    manifoldTangentialBasis = torch.matmul(diffeoJacobis, sphereTangentialBasis).squeeze(-1)
    manifoldUnitTangentialBasis = torch.nn.functional.normalize(manifoldTangentialBasis, dim=-1)
    return manifoldUnitTangentialBasis

def getManifoldNormalsFromSphereCoords(sphereCoords: torch.Tensor, manifoldMap):
    manifoldTangentBasis = getManifoldTangentBasisFromSphereCoords(sphereCoords, manifoldMap)
    tan0 = manifoldTangentBasis[...,0,:]
    tan1 = manifoldTangentBasis[...,1,:]
    return torch.nn.functional.normalize(torch.linalg.cross(tan0, tan1), dim=-1)

def getManifoldNormalsFromFlatCoords(flatCoords: torch.Tensor, manifoldMap):
    sphereCoords = getSphereCoordsFromFlatCoords(flatCoords)
    return getManifoldNormalsFromSphereCoords(sphereCoords, manifoldMap)

def getManifoldProjectionsFromSphereCoords(sphereCoords: torch.Tensor, manifoldMap):
    normals = getManifoldNormalsFromSphereCoords(sphereCoords, manifoldMap)
    return torch.eye(3).repeat(list(sphereCoords.shape[0:-1]) + [1, 1]) - torch.matmul(normals.unsqueeze(-1), normals.unsqueeze(-2))

def getFirstFundamentalForm(flatCoords: torch.Tensor, manifoldMap):

    def fullParametrization(coordsIn):
        return getManifoldCoordsFromFlatCoords(coordsIn, manifoldMap)

    targetJacobiShape = list(flatCoords.shape[0:-1]) + [3, 2]
    singularCoords = torch.flatten(flatCoords, start_dim=0, end_dim=-2)
    fullJacobis = torch.reshape(torch.vmap(torch.func.jacfwd(fullParametrization))(singularCoords), targetJacobiShape)
    firstFundForms = torch.matmul(torch.transpose(fullJacobis, dim0=-2, dim1=-1), fullJacobis)

    E = firstFundForms[...,0,0]
    F = firstFundForms[...,0,1]
    G = firstFundForms[...,1,1]

    return E, F, G

def getSecondFundamentalForm(flatCoords: torch.Tensor, manifoldMap):

    def fullParametrization(coordsIn):
        return getManifoldCoordsFromFlatCoords(coordsIn, manifoldMap)

    normals = getManifoldNormalsFromFlatCoords(flatCoords, manifoldMap)

    targetHessianShape = list(flatCoords.shape[0:-1]) + [3, 2, 2]
    singularCoords = torch.flatten(flatCoords, start_dim=0, end_dim=-2)
    fullHessian = torch.reshape(torch.vmap(torch.func.jacfwd(torch.func.jacfwd(fullParametrization)))(singularCoords), targetHessianShape)
    fullHessian = fullHessian.to(torch.float32)
    
    rUU = fullHessian[...,:,0,0]
    rUV = fullHessian[...,:,0,1]
    rVV = fullHessian[...,:,1,1]

    L = torch.matmul(rUU.unsqueeze(-2), normals.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    M = torch.matmul(rUV.unsqueeze(-2), normals.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    N = torch.matmul(rVV.unsqueeze(-2), normals.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    return L, M, N

def getCurvaturesFromSphereCoords(sphereCoords: torch.Tensor, manifoldMap):
    flatCoords = getFlatCoordsFromSphereCoords(sphereCoords)
    E, F, G = getFirstFundamentalForm(flatCoords, manifoldMap)
    L, M, N = getSecondFundamentalForm(flatCoords, manifoldMap)

    gaussianCurvature = (L * N - M * M) / (E * G - F * F)
    meanCurvature = (L * G - 2. * M * F + N * E) / (2 * (E * G - F * F))

    return gaussianCurvature, meanCurvature

def getGeometricPotential(sphereCoords: torch.Tensor, manifoldMap):
    gaussianCurvature, meanCurvature = getCurvaturesFromSphereCoords(sphereCoords, manifoldMap)
    return (torch.pow(meanCurvature, 2.) - gaussianCurvature).to(torch.float32)


def getTotalCurvature(sphereCoords: torch.Tensor, manifoldMap):
    flatCoords = getFlatCoordsFromSphereCoords(sphereCoords)
    E, F, G = getFirstFundamentalForm(flatCoords, manifoldMap)
    L, M, N = getSecondFundamentalForm(flatCoords, manifoldMap)

    meanCurvature = (L * G - 2. * M * F + N * E) / (2 * (E * G - F * F))

    return 2. * meanCurvature.to(torch.float32)