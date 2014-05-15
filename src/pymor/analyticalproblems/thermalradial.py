# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import product
import numpy as np

from pymor.analyticalproblems.elliptic import EllipticProblem
from pymor.core import Unpicklable, inject_sid
from pymor.domaindescriptions import RectDomain
from pymor.functions import GenericFunction, ConstantFunction
from pymor.parameters import CubicParameterSpace, ProjectionParameterFunctional


class TRBP(EllipticProblem, Unpicklable):
    '''Analytical description of a 2D thermal block diffusion problem.

    This problem is to solve the elliptic equation ::

      - ∇ ⋅ [ d(x, μ) ∇ u(x, μ) ] = f(x, μ)

    on the domain [0,1]^2 with Dirichlet zero boundary values. The domain is
    partitioned into nx x ny blocks and the diffusion function d(x, μ) is
    constant on each such block (i,j) with value μ_ij. ::

           ----------------------------
           |        |        |        |
           |  μ_11  |  μ_12  |  μ_13  |
           |        |        |        |
           |---------------------------
           |        |        |        |
           |  μ_21  |  μ_22  |  μ_23  |
           |        |        |        |
           ----------------------------

    The Problem is implemented as an |EllipticProblem| with the
    characteristic functions of the blocks as `diffusion_functions`.

    Parameters
    ----------
    num_blocks
        n
    parameter_range
        A tuple (μ_min, μ_max). Each |Parameter| component μ_ij is allowed
        to lie in the interval [μ_min, μ_max].
    rhs
        The |Function| f(x, μ).
    '''

    def __init__(self, num_blocks=(3), parameter_range=(0.1, 1), rhs=ConstantFunction(dim_domain=2)):



        domain = RectDomain()
        parameter_space = CubicParameterSpace({'diffusion': num_blocks}, *parameter_range)
        dx = 1 / num_blocks[0]

        r = 1 / (num_blocks[0] * np.sqrt(2))

        #dy = 1 / num_blocks[1]

        # creating the id-string once for every diffusion function reduces the size of the pickled sid
        diffusion_function_id = str(ThermalRadialBlockProblem) + '.diffusion_function'

        def diffusion_function_factory(x):
            func = lambda X: (1. * (X[..., 0] >= (x * r) * (X[..., 0] < (x + 1) * r)))


            inject_sid(func, diffusion_function_id, x, r)

            return GenericFunction(func, dim_domain=2, name='diffusion_function_{}_{}'.format(x))

        def parameter_functional_factory(x):
            return ProjectionParameterFunctional(component_name='diffusion',
                                                 component_shape=(num_blocks[0]),
                                                 coordinates=(x),
                                                 name='diffusion_{}_{}'.format(x))

        diffusion_functions = tuple(diffusion_function_factory(x)
                                    for x, y in xrange(num_blocks[0]))
        parameter_functionals = tuple(parameter_functional_factory(x)
                                      for x, y in xrange(num_blocks[0]))

        super(ThermalRadialBlockProblem, self).__init__(domain, rhs, diffusion_functions, parameter_functionals,
                                                  name='ThermalBlock')
        self.parameter_space = parameter_space
        self.parameter_range = parameter_range
        self.num_blocks = num_blocks
