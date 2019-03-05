import ase
import numpy as np
from .helpers import *
from toolz.curried import curry, pipe


@curry
def get_scaled_positions(coords, cell, pbc, wrap=True):
    """Get positions relative to unit cell.
    If wrap is True, atoms outside the unit cell will be wrapped into
    the cell in those directions with periodic boundary conditions
    so that the scaled coordinates are between zero and one."""

    fractional = np.linalg.solve(cell.T, coords.T).T

    if wrap:
        for i, periodic in enumerate(pbc):
            if periodic:
                fractional = np.mod(fractional, 1.0)
    return fractional


@curry
def get_real_positions(coords, cell):
    """Return real space coordinates
    given fractional coordiantes and
    cell parameters"""
    return np.dot(cell.T, coords.T).T


@curry
def AtomCenters(coords, box, len_pixel):
    """
    Args: Coordinates of all atoms [ndArray], dimension of box, pixel size

    returns: Atom Center location voxels
    """

    atom_centers = np.zeros(coords.shape)
    for dim in range(3):
        atom_centers[:,dim] = pipe(coords[:,dim],
                                   lambda x: np.around(x*len_pixel))
    return atom_centers


@curry
def grid_maker_fft(atom, len_pixel=10, atomSi=1.35, atomOx=1.35, full=False):
    dgnls = atom.cell.diagonal()
    coords = pipe(atom,
                  lambda x: x.get_positions(),
                  lambda x: np.mod(x, dgnls),
                  lambda x: x - x.min(axis=0))
    box_dim  = np.ceil((coords.max(axis=0)) * len_pixel).astype(int) + 1

    atom_ids = np.array(atom.get_chemical_symbols())
    idx_Ox = np.where(atom_ids == 'O')[0]
    idx_Si = np.where(atom_ids == 'Si')[0]

    atom_centers = AtomCenters(coords, box_dim, len_pixel)
    x , y, z = [atom_centers[:,dim].astype(int) for dim in range(3)]

    S_Ox, S_Si = np.zeros(box_dim), np.zeros(box_dim)

    S_Ox[x[idx_Ox], y[idx_Ox], z[idx_Ox]] = 1
    S_Si[x[idx_Si], y[idx_Si], z[idx_Si]] = 1

    if full:
        scaler = [len_pixel * atomSi.shape[0]] * 3
    else:
        scaler = 0.0
    scaled_box_dim = (box_dim + scaler)

    if np.isscalar(atomSi):
        atomSi = sphere(atomSi * len_pixel)

    if np.isscalar(atomOx):
        atomOx = sphere(atomOx * len_pixel)

    atomSi = padder(atomSi, scaled_box_dim)
    atomOx = padder(atomOx, scaled_box_dim)

    S_Ox = padder(S_Ox, scaled_box_dim)
    S_Si = padder(S_Si, scaled_box_dim)

    S_Ox = (imfilter(S_Ox, atomOx) > 1e-1) * 1
    S_Si = (imfilter(S_Si, atomSi) > 1e-1) * 1

    S    = ((S_Ox + S_Si) < 1e-1) * 1

    return S, S_Ox, S_Si, box_dim

@curry
def grid_maker_edt(atom, len_pixel=10, r_Si=1.35, r_Ox=1.35, full=False):
    dgnls = atom.cell.diagonal()
    coords = pipe(atom,
                  lambda x: x.get_positions(),
                  lambda x: np.mod(x, dgnls),
                  lambda x: x - x.min(axis=0))
    box_dim  = np.ceil((coords.max(axis=0)) * len_pixel).astype(int) + 1

    atom_ids = np.array(atom.get_chemical_symbols())
    idx_Ox = np.where(atom_ids == 'O')[0]
    idx_Si = np.where(atom_ids == 'Si')[0]

    atom_centers = AtomCenters(coords, box_dim, len_pixel)
    x , y, z = [atom_centers[:,dim].astype(int) for dim in range(3)]

    S_Ox, S_Si = np.ones(box_dim), np.ones(box_dim)

    S_Ox[x[idx_Ox], y[idx_Ox], z[idx_Ox]] = 0
    S_Si[x[idx_Si], y[idx_Si], z[idx_Si]] = 0

    if full:
        scaler = [len_pixel * (2*r_Si+1)] * 3
        scaled_box_dim = (box_dim + scaler)
        S_Ox = padder(S_Ox, scaled_box_dim, 1)
        S_Si = padder(S_Si, scaled_box_dim, 1)

    S_Ox = transform_edt(S_Ox.astype(np.uint8)) / len_pixel
    S_Si = transform_edt(S_Si.astype(np.uint8)) / len_pixel

    S_Ox = (S_Ox < r_Ox) * 1
    S_Si = (S_Si < r_Si) * 1

    S    = ((S_Ox + S_Si) < 0.1) * 1

    return S, S_Ox, S_Si, box_dim
