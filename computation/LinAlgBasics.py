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
    vNorm = torch.linalg.norm(v).item()
    paddedV = torch.zeros(n) 
    paddedV[n - v.shape[0]:] = v
    target = torch.zeros(n)
    target[n - v.shape[0]] = np.sign(v[0].item()) * vNorm
    householderVector = paddedV - target
    householderNorm = torch.linalg.norm(householderVector)
    householderVector = (householderVector / householderNorm).unsqueeze(1)
    return torch.eye(n) - 2 * torch.matmul(householderVector, torch.transpose(householderVector, dim0=0, dim1=1))

def tridiagonalizeSkewSymmetricMatrix(A: torch.Tensor):
    matSize = A.shape[0]
    tridiagMat = A
    for column_index in range(matSize-2):
        householderV = tridiagMat[column_index+1:matSize,column_index].squeeze()
        Q = getHouseholderTransform(householderV, matSize)
        tridiagMat = torch.matmul(torch.matmul(Q, tridiagMat), torch.transpose(Q, dim0=0, dim1=1))
    return tridiagMat

#Note: THis function expects 2D tensors (= Matrix). Remember to use vmap for batches.
def getPfaffian(A: torch.Tensor):
    matSize = A.shape[0]
    if matSize % 2 == 1:
        return torch.tensor([0.0])
    triDiag = tridiagonalizeSkewSymmetricMatrix(A)
    result = torch.tensor([1.0])
    for column_index in range(int(matSize/2)):
        result[0] = result[0] * triDiag[2 * column_index, 2 * column_index + 1]
    return result
