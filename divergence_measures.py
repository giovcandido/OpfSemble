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

def kl_divergence_matrix(preds):
    '''
    Function that performs the Kullback-Lieber divergence between each pair of samples in 'preds'

    Parameters
    ----------
    preds : array
        Array with N rows and M columns.

    Returns
    -------
    kl_array: array
        An array with size NxN with the KL divergence between the pairs of samples in 'preds'
    '''
    
    # calculate the kl divergence
    def kl_divergence(p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))
        
    kl_array = np.zeros((preds.shape[0],preds.shape[0]))
    for i in range(preds.shape[0]):
        for j in range(preds.shape[0]):
            # Getting the maximum value between the two arrays
            max_ = max(np.max(preds[i,:]),np.max(preds[j,:]))
            c1, _ = np.histogram(preds[i,:], bins=np.arange(max_+2))
            c2, _ = np.histogram(preds[j,:], bins=np.arange(max_+2))
            p = c1 / preds.shape[1]
            q = c2 / preds.shape[1]
            kl_array[i,j] = kl_divergence(p,q)

    un = np.unique(kl_array)

    if (len(un) > 1):
        # Correction of infinity values
        inf_value = 10

        if (un[-2] == 0): # [0, inf]
            kl_array[kl_array == np.inf] = inf_value
        else: # [0, value, inf]
            kl_array[kl_array == np.inf] = un[-2] * inf_value
    
    return kl_array    