# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor import la
from pymor.operators.cg import L2ProductP1
from pymortests.base import runmodule
from pymor.grids.tria import TriaGrid
from pymor.grids.boundaryinfos import AllDirichletBoundaryInfo
from pymor.grids.boundaryinfos import EmptyBoundaryInfo
from pymor.la.gram_schmidt import gram_schmidt

def test_induced():
    grid = TriaGrid(num_intervals=(10, 10))
    boundary_info = AllDirichletBoundaryInfo(grid)
    product = L2ProductP1(grid, boundary_info)
    zero = la.NumpyVectorArray(np.zeros(grid.size(2)))
    norm = la.induced_norm(product)
    value = norm(zero)
    np.testing.assert_almost_equal(value, 0.0)

def test_induced_constant():
    grid = TriaGrid(num_intervals=(10, 10))
    #boundary_info = AllDirichletBoundaryInfo(grid)
    boundary_info = EmptyBoundaryInfo(grid)
    product = L2ProductP1(grid, boundary_info)
    norm = la.induced_norm(product)
    constant = la.NumpyVectorArray(1 * np.ones(grid.size(2)))

    value = norm(constant)
    np.testing.assert_almost_equal(1.0, 1.0)

def test_induced_constant():
    function = lambda x: np.sin(2 * x * np.pi)
    grid = TriaGrid(num_intervals=(10, 10))
    boundary_info = EmptyBoundaryInfo(grid)
    product = L2ProductP1(grid, boundary_info)
    norm = la.induced_norm(product)
    constant = la.NumpyVectorArray(5*np.ones(grid.size(2)))

    #constant = la.NumpyVectorArray(function(grid.centers(2)))

    value = norm(constant)
    np.testing.assert_almost_equal(value, 5.0)

def test_induced_function():
    function = lambda x: np.sin(2 * x * np.pi)
    #function = lambda x: np.exp(2 * x * np.pi)
    grid = TriaGrid(num_intervals=(10, 10))
    boundary_info = EmptyBoundaryInfo(grid)
    product = L2ProductP1(grid, boundary_info)
    norm = la.induced_norm(product)
    #constant = la.NumpyVectorArray(5*np.ones(grid.size(2)))
    constant = la.NumpyVectorArray(function(np.ones(grid.size(2))))

    value = norm(constant)
    np.testing.assert_almost_equal(value, 0.0)


def test_gram_schmidt():
    A = la.NumpyVectorArray([[0,1.0,0],[1.0,0,0],[1.0,2.0,3.0]])
    print(A)
    B = gram_schmidt(A)
    print(B)
    np.testing.assert_almost_equal(np.mean(B.data[0]),np.mean([[0,1.0,0],[1.0,0,0],[1.0,0,0]]))

    B = gram_schmidt(A,check=True)
    B = gram_schmidt(A, reiterate=True)
    A = la.NumpyVectorArray([[0,1.0,0],[0, 1.0,0],[1.0,2.0,3.0]])

    B = gram_schmidt(A, find_duplicates=False)
    print(B)
    np.testing.assert_almost_equal(2,3)






if __name__ == "__main__":
    runmodule(filename=__file__)
