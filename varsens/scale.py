"""A collection of helper scaling functions that map low discrepency sequences into desired ranges.
"""

import numpy

def linear(points, lower_bound, upper_bound):
    """Linearly scale a numpy array from the domain [0,1] to the range [lower,upper]

    Parameters
    ----------
    points : numpy.array
        numeric data to map in the domain [0,1] (usually from a low-discrepency sequence)
    lower_bound : numpy.array
        The lower bounds of mapping. Must be same shape as points
    upper_bound : numpy.array
        The upper bounds of mapping. Must be same shape as points
        a numpy array of the scaled points

    Returns
    -------
    scaled points : numpy.array

    Examples
    ________

        >>> from numpy import *
        >>> from varsens import *
        >>> scale.linear(array([0.5]*3), array([-100, -10, 1000]), array([100, 20, 2000]))
        array([    0.,     5.,  1500.])
        >>>

    """
    return points*(upper_bound - lower_bound) + lower_bound

def power(points, lower_bound, upper_bound):
    """Power (exponentiate) scale a numpy array from the domain [0,1] to the range [lower,upper]

    Parameters
    ----------
    points : numpy.array
        numeric data to map in the domain [0,1] (usually from a low-discrepency sequence)
    lower_bound : numpy.array
        The lower bounds of mapping. Must be same shape as points, and be all positive floats
    upper_bound : numpy.array
        The upper bounds of mapping. Must be same shape as points
        a numpy array of the scaled points

    Returns
    -------
    scaled points : numpy.array

    Examples
    ________

        >>> from numpy import *
        >>> from varsens import *
        >>> scale.power(array([0.5]*3), array([10, 100, 1000]), array([1000, 200, 2500]))
        array([  100.        ,   141.42135624,  1414.21356237])
        >>>

    """
    return lower_bound*((upper_bound / lower_bound)**points)

def percentage(points, reference, percentage=50.0):
    """Linearly scale a numpy array from the domain [0,1] to the reference +/- percentage

    Parameters
    ----------
    points : numpy.array
        numeric data to map in the domain [0,1] (usually from a low-discrepency sequence)
    reference : numpy.array
        The reference in the middle of the desired range
    percentage : numpy.array
        A number in the range (0, Inf). Never go above 100.0 if you desire the lower bound to be above 0.0.

    Returns
    -------
    scaled points : numpy.array

    Examples
    ________

        >>> from numpy import *
        >>> from varsens import *
        >>> scale.percentage(array([0.333]*3), array([1, 10, 1000]), 50.0)
        array([   0.833,    8.33 ,  833.   ])
        >>>

    """
    diff = percentage * reference / 100.0
    return linear(points, reference-diff, reference+diff)

def magnitude(points, reference, orders=3.0, base=10.0):
    """Power scale a numpy array from the domain [0,1] to a power range given in orders of magnitude

    Parameters
    ----------
    points : numpy.array
        numeric data to map in the domain [0,1] (usually from a low-discrepency sequence)
    reference : numpy.array
        The reference in the logrithmic middle of the desired range
    orders : float
        A single number indicating the +/- orders of magnitude to apply
    base : float
        A single number that is the magnitude scale

    Returns
    -------
    scaled points : numpy.array

    Examples
    ________

        >>> from numpy import *
        >>> from varsens import *
        >>> scale.magnitude(array([0.333]*3), array([1, 10, 1000]))
        array([  0.09954054,   0.99540542,  99.54054174])
        >>>

    """
    factor = base ** orders
    return power(points, reference / factor, reference * factor)

