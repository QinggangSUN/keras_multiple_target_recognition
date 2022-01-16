# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 22:40:59 2020

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class ParameterError(Error):
    """Exception raised for errors in the Parameter of a function."""
    pass


class FunctionCallError(Error):
    """Exception raised for errors in the call of a function."""
    pass
