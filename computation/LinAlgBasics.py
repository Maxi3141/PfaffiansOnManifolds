import numpy as np
import torch

def getHouseholderTransformNumpy(v: np.typing.ArrayLike, n: int):
    v_norm = np.linalg.norm(v)
    padded_v = np.expand_dims(np.pad(v, (n - v.size, 0), 'constant'), axis=1)
    target = np.zeros((n,1))
    target[n - v.size] = v_norm * np.sign(v[0])
    householder_vector = padded_v - target
    householder_norm = np.linalg.norm(householder_vector)
    householder_vector = householder_vector / householder_norm
    return np.identity(n) - 2 * householder_vector @ np.transpose(householder_vector)

def tridiagonalizeSkewSymmetricMatrixNumpy(A: np.typing.ArrayLike):
    mat_size = np.shape(A)[0]
    tridiag_mat = A
    for column_index in range(mat_size-2):
        householder_v = tridiag_mat[column_index+1:mat_size,column_index]
        Q = getHouseholderTransformNumpy(householder_v, mat_size)
        tridiag_mat = Q @ tridiag_mat @ Q.T
    return tridiag_mat

def getPfaffianNumpy(A: np.typing.ArrayLike):
    mat_size = np.shape(A)[0]
    if mat_size % 2 == 1:
        return 0
    triDiag = tridiagonalizeSkewSymmetricMatrixNumpy(A)
    result = 1
    for column_index in range(int(mat_size/2)):
        result *= triDiag[2 * column_index, 2 * column_index + 1]
    return result

def getHouseholderTransform(v: torch.Tensor, n: int):
    vNorm = torch.linalg.norm(v, ord=2., dim=-1, keepdim=True)
    paddedV = torch.cat([torch.zeros(list(v.shape[:-1]) + [n - v.shape[-1]]), v], dim=-1)
    target = torch.cat([torch.zeros(list(v.shape[:-1]) + [n - v.shape[-1]]), torch.sign(v[...,:1]) * vNorm, torch.zeros(list(v.shape[:-1]) + [v.shape[-1] - 1])], dim=-1)
    householderVector = torch.nn.functional.normalize(paddedV - target, p=2., dim=-1)
    invResult = 2. * torch.matmul(householderVector.unsqueeze(-1), householderVector.unsqueeze(-2))
    return torch.eye(n).repeat(invResult[...,:1,:1].shape) - invResult

def tridiagonalizeSkewSymmetricMatrix(A: torch.Tensor):
    matSize = A.shape[-1]
    tridiagMat = A
    for column_index in range(matSize-2):
        householderV = tridiagMat[...,column_index+1:matSize,column_index]
        Q = getHouseholderTransform(householderV, matSize)
        tridiagMat = torch.matmul(torch.matmul(Q, tridiagMat), torch.transpose(Q, dim0=-2, dim1=-1))
    return tridiagMat

def getPfaffian(A: torch.Tensor):
    matSize = A.shape[-1]
    if matSize % 2 == 1:
        return torch.zeros(A.shape[:-2])
    triDiag = tridiagonalizeSkewSymmetricMatrix(A)
    result = torch.prod(torch.diagonal(triDiag[...,:,1::2][...,::2,:], offset=0, dim1=-2, dim2=-1), dim=-1)
    return result

