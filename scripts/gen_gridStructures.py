import glob
import multiprocessing as mp
import os
import time

import ase.io as aio
import click
import numpy as np
import scipy.io as sio
from toolz.curried import curry, pipe

import atommks.grid_generator as gen
import atommks.porosity as pore
from atommks.helpers import write2vtk


@curry
def structure_maker(
    fname,
    radii={"Si": 1.35, "O": 1.35, "H": 1.0},
    len_pixel=10,
    rep=[1, 1, 1],
    save_dir="",
):
    """
    Saves the voxelized structure in matfile format
    """
    try:

        cif = pipe(
            fname,
            lambda x: os.path.split(x)[-1][:-4],
            lambda x: os.path.join(save_dir, x),
        )

        atom = aio.read(fname).repeat(rep)
        S = gen.grid_maker(
            atom, len_pixel=10, atomic_radii=radii, full=False, fft=True
        )[0]

        padval = ((1, 1), (1, 1), (0, 0))
        S_dgrid = pipe(
            S,
            lambda s: np.pad(s, padval, "constant", constant_values=0),
            lambda s: pore.dgrid(s, len_pixel),
        )
        sio.savemat("%s_dgrid.mat" % cif, {"s": S_dgrid})
        write2vtk(S, "%s_pore.vtk" % cif)
        print(cif)
    except Exception as err:
        print("Exception for file : %s" % (fname), err)


@click.command()
@click.option(
    "--input-dir",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input directory containing CIF files.",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    help="Output directory to save processed files.",
)
@click.option(
    "--len-pixel",
    "-l",
    default=10,
    type=int,
    help="Length of pixel grid (default is 10).",
)
@click.option(
    "--rep",
    "-r",
    default="1,1,1",
    help='Repeat the structure (default is "1,1,1").',
)
@click.option(
    "--processes",
    "-p",
    default=1,
    type=int,
    help="Number of parallel processes to use (default is 1).",
)
def process_structures(input_dir, output_dir, len_pixel, rep, processes):
    """
    Command-line tool to process structures in CIF format with parallel processing. # noqa: E501
    """
    rep = [
        int(x) for x in rep.split(",")
    ]  # Convert the repeat argument to a list of integers

    print(
        f"Processing structures in directory: {input_dir}, with {processes} processes and saving to {output_dir}. repeat={rep}, len_pixel={len_pixel}"  # noqa: E501
    )

    flist = sorted(glob.glob(os.path.join(input_dir, "*.cif")))

    if not flist:
        print(f"No CIF files found in directory: {input_dir}")
        return

    print(f"Found {len(flist)} CIF files in the directory.")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    func = structure_maker(len_pixel=len_pixel, rep=rep, save_dir=output_dir)
    with mp.Pool(processes=processes) as pool:
        pool.map(func, flist)

    print(f"Processing complete. Files saved to {output_dir}.")


if __name__ == "__main__":
    strt = time.time()
    process_structures()  # Click command entry point
    end = time.time()
    print(f"Total processing time: {end - strt} seconds.")
