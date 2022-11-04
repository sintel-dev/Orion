import operator

def detect_anomalies(X, thres, op, func=None, dim=None, arg=None):
    """
    The function detects anomalies that are flagged through one liners that can be constructed using parameters like func, hres, arg

    Args:
        X (ndarray): 
            N-dimnsional value sequence to iterate over.
        thres (float):
            Integer used to indicate the threshold of the function
        op (str):
            String indicating the operator used to compare with the threshold. Possible values are '<', '>', '<=', '>=', '=='
        func (str):
            String indicating the function/ computation of the one-liner. Possible values for func are 'diff', 'movstd'
        dim (int):
            Integer indicating the dimension number for a multi-dimensional dataset
        arg (float):
            Any argument that might be used as a parameter in 'func'. As of now, only movstd has a parameter. Therefore arg indicates the window size for movstd
        

    Returns:
        list:
            integers indicating the indices of the dataset that were flagged by the one-liner generated

    """

    if (len(X) == 0 or op == None or thres == None):
        return []
    
    if dim != None:
        if dim in X.columns.values:
            value_list = X[dim]
            X['value'] = X[dim]
    value_list = X['value']
    
    ops = {'>': operator.gt,
           '<': operator.lt,
           '>=': operator.ge,
           '<=': operator.le,
           '==': operator.eq}
    

    if func == "diff":
        a = np.diff(value_list)
        return [idx for idx in range(len(a)) if ops[op](a[idx],  thres)]
    
    elif func == "movstd":
        a = value_list.rolling(window=arg).std()
        return [idx for idx in range(len(a)) if ops[op](a[idx],  thres)]
    
    elif func == None:
        return [idx for idx in range(len(value_list)) if ops[op](value_list[idx], thres)]
    
    
    return []