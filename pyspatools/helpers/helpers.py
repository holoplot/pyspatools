def lin_map(x, in_min, in_max, out_min, out_max):
    """
    Linear mapping a scala
    Parameters
    ----------
    x : float
        input value
    in_min : float
        input's minimum value
    in_max : float
        input's maximum value
    out_min : float
        output's minimum value
    out_max : float
        output's maximum value

    Returns
    -------
    float
        mapped output
    """
    return (x - in_min) / (in_max - in_min) * (out_max - out_min) + out_min

