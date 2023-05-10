import numpy as np

def __coefficients(preds):
    '''
    Function that determines when two classifiers agree or disagree in their predictions

    Parameters
    ----------
    preds: array
        An Mx2 array with the 'M' predictions of two classifiers, where M is the number of samples used for predictions

    Returns
    -------
    a,b,c,d: bool
        Indications containing the four types of disagreements between A and B
    '''

    A = np.asarray(preds[:, 0], dtype=bool)
    B = np.asarray(preds[:, 1], dtype=bool)

    a = np.sum(A * B)           # A right, B right
    b = np.sum(~A * B)          # A wrong, B right
    c = np.sum(A * ~B)          # A right, B wrong
    d = np.sum(~A * ~B)         # A wrong, B wrong

    return a, b, c, d

def disagreement(preds, i,j):
    '''
    Function that determines the disagreement between two classifiers 'i' and 'j'

    Parameters
    ----------
    preds: array
        An MxN array with the 'M' predictions of 'N' classifiers, where M is the number of samples used for predictions, and N is the number of classifiers
    i,j: int
        Indices of the classifiers in the array of predictions

    Returns
    -------
    disagreement_rate: float
        The disagreement rate between 'i' and 'j'
    '''

    L = preds.shape[1]
    a, b, c, d = __coefficients(preds[:, [i, j]])
    return float(b + c) / (a + b + c + d)

def disagreement_matrix(preds):
    '''
    Function that computes the disagreement matrix among the classifiers' predictions

    Parameters
    ----------
    preds: array
        An MxN array with the 'M' predictions of 'N' classifiers, where M is the number of samples used for predictions, and N is the number of classifiers

    Returns
    -------
    res: array
        The disagreement matrix with N rows and N cols (NxN), where N is the number of classifiers
    '''

    res = np.zeros((preds.shape[1], preds.shape[1]))
    for i in range(preds.shape[1]):
        for j in range(i, preds.shape[1]):
            res[i, j] = disagreement(preds, i, j)
            res[j, i] = res[i, j]
    return res    

def paired_q(preds, i, j):
    '''
    Function that computes the Yule's Q test between two classifiers 'i' and 'j'

    Parameters
    ----------
    preds: array
        An MxN array with the 'M' predictions of 'N' classifiers, where M is the number of samples used for predictions, and N is the number of classifiers
    i,j: int
        Indices of the classifiers in the array of predictions

    Returns
    -------
    yule_rate: float
        The disagreement rate between 'i' and 'j'        
    '''

    L = preds.shape[1]
    # div = np.zeros((L * (L - 1)) // 2)
    a, b, c, d = __coefficients(preds[:, [i, j]])
    return float(a * d - b * c) / ((a * d + b * c) + 10e-24)

def paired_q_matrix(preds):
    '''
    Function that computes the Yule's Q test matrix among the classifiers' predictions

    Parameters
    ----------
    preds: array
        An MxN array with the 'M' predictions of 'N' classifiers, where M is the number of samples used for predictions, and N is the number of classifiers

    Returns
    -------
    res: array
        The disagreement matrix with N rows and N cols (NxN), where N is the number of classifiers
    '''

    res = np.zeros((preds.shape[1], preds.shape[1]))
    for i in range(preds.shape[1]):
        for j in range(i, preds.shape[1]):
            res[i, j] = paired_q(preds, i, j)
            res[j, i] = res[i, j]
    return res    