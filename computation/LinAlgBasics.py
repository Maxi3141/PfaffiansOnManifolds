import numpy as np

def getHouseholderTransform(v: np.typing.ArrayLike, n: int):
    v_norm = np.linalg.norm(v)
    padded_v = np.expand_dims(np.pad(v, (n - v.size, 0), 'constant'), axis=1)
    target = np.zeros((n,1))
    target[n - v.size] = v_norm * np.sign(v[0])
    householder_vector = padded_v - target
    householder_norm = np.linalg.norm(householder_vector)
    householder_vector = householder_vector / householder_norm
    return np.identity(n) - 2 * householder_vector @ np.transpose(householder_vector)

def tridiagonalizeSkewSymmetricMatrix(A: np.typing.ArrayLike):
    mat_size = np.shape(A)[0]
    tridiag_mat = A
    for column_index in range(mat_size-2):
        householder_v = tridiag_mat[column_index+1:mat_size,column_index]
        Q = getHouseholderTransform(householder_v, mat_size)
        tridiag_mat = Q @ tridiag_mat @ Q.T
    return tridiag_mat

def getPfaffian(A: np.typing.ArrayLike):
    mat_size = np.shape(A)[0]
    if mat_size % 2 == 1:
        return 0
    triDiag = tridiagonalizeSkewSymmetricMatrix(A)
    result = 1
    for column_index in range(int(mat_size/2)):
        result *= triDiag[2 * column_index, 2 * column_index + 1]
    return result
