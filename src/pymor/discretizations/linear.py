from __future__ import absolute_import, division, print_function, unicode_literals

import copy

import numpy as np
from scipy.sparse.linalg import bicg
from scipy.sparse import issparse

from pymor.core import BasicInterface
from pymor.core.cache import Cachable, cached, DEFAULT_DISK_CONFIG
from pymor.tools import dict_property
from pymor.domaindescriptions import BoundaryType
from pymor.parameters import Parametric


class StationaryLinearDiscretization(BasicInterface, Parametric, Cachable):

    operator = dict_property('operators', 'operator')
    rhs = dict_property('operators', 'rhs')

    def __init__(self, operator, rhs, solver=None, visualizer=None):
        Cachable.__init__(self, config=DEFAULT_DISK_CONFIG)
        self.operators = {'operator': operator, 'rhs':rhs}
        self.build_parameter_type(inherits={'operator':operator, 'rhs':rhs})

        def default_solver(A, RHS):
            if issparse(A):
                U, info = bicg(A, RHS)
            else:
                U = np.linalg.solve(A, RHS)
            return U
        self.solver = solver or default_solver

        if visualizer is not None:
            self.visualize = visualizer

        self.solution_dim = operator.range_dim

    def copy(self):
        c = copy.copy(self)
        c.operators = c.operators.copy()
        Cachable.__init__(c)
        return c

    @cached
    def solve(self, mu={}):
        mu = self.parse_parameter(mu)
        self.logger.info('Solving for {} ...'.format(mu))

        A = self.operator.matrix(self.map_parameter(mu, 'operator'))
        if A.size == 0:
            return np.zeros(0)
        RHS = np.squeeze(self.rhs.matrix(self.map_parameter(mu, 'rhs')))
        if RHS.ndim == 0:
            RHS = RHS[np.newaxis]

        return self.solver(A, RHS)
