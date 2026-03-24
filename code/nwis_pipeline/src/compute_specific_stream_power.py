


def specific_stream_power(gamma: 9800, Q: float, S: float, w: float):
    '''
    Compute specific stream power.
        gamma       unit weight of water (kN/m3) = 9800
        Q           discharge (m3/s)
        S           energy slope (often channel slope) (m/m)
        w           bankfull width (m)
    '''
    
    return (gamma * Q * S) / w
