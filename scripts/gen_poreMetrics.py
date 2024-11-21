import glob
import multiprocessing as mp
import os
import time

import click
import numpy as np
import scipy.io as sio
from toolz.curried import curry, pipe

import atommks.porosity as pore


@curry
def poreStructureMaker(fname, save_dir="", r_probe=0.1, len_pixel=10):
    try:
        cif = pipe(
            fname,
            lambda x: os.path.split(x)[-1].split("_dgrid")[0],
            lambda x: os.path.join(save_dir, x),
        )

        S = sio.loadmat(fname)["s"]

        pld = pore.get_pld(S)
        lcd = pore.get_lcd(S)

        # generates probe accessible pore region
        S_1 = (
            pore.gen_cleanPore(
                S, r_probe=r_probe, r_min=2.5, len_pixel=len_pixel
            )
            > 0
        ) * 1

        # generates medial axis of the accessible pore region
        S_2 = pipe(
            S_1,
            lambda x: np.pad(
                x,
                pad_width=((0, 0), (0, 0), (len_pixel, len_pixel)),
                mode="constant",
                constant_values=1,
            ),
            lambda x: pore.gen_medialAxis(x)[:, :, len_pixel:-len_pixel],
        )

        # Prunes medial axis to return, only the paths connecting opposing surfaces # noqa: E501
        S_3, paths = pore.gen_throughPath(S_2, depth=1)

        # Number of independant transport channels in the structure
        n_paths = len(pore.return_labelled(S_1)[-1])

        # accessible surface area
        asa = pore.get_asa(S_1, len_pixel=10)

        # accessile volume
        av = np.count_nonzero(S_1) * (1 / len_pixel) ** 3

        # pore size distribution
        psd = S[S_2 == 1]

        # dimensions of the structure
        dim = np.asarray(S.shape) / len_pixel

        # save all computed data as a matfile
        sio.savemat(
            "%s_pore" % cif,
            {
                "pld": pld,
                "lcd": lcd,
                "n_paths": n_paths,
                "asa": asa,
                "av": av,
                "dim": dim,
                "paths": paths,
                "psd": psd,
                "len_pixel": len_pixel,
            },
        )

        # print(cif, pld, lcd, asa, av, n_paths, np.mean(paths), np.mean(psd))

    except Exception as err:
        print("Exception for file : %s" % (fname), err)


@click.command()
@click.option(
    "--input-folder",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input folder containing MAT files.",
)
@click.option(
    "--output-folder",
    "-o",
    required=True,
    type=click.Path(),
    help="Output folder to save processed files.",
)
@click.option(
    "--r-probe",
    "-r",
    default=1.0,
    type=float,
    help="Probe radius (default is 1.0).",
)
@click.option(
    "--len-pixel",
    "-l",
    default=10,
    type=int,
    help="Number of voxels per angstrom (default is 10).",
)
@click.option(
    "--processes",
    "-p",
    default=2,
    type=int,
    help="Number of parallel processes (default is 2).",
)
def prll(input_folder, output_folder, r_probe, len_pixel, processes):
    """
    Command-line tool to process MAT files for pore structure analysis.
    """
    # Get list of MAT files
    input_fnames = os.path.join(input_folder, "*_dgrid.mat")
    flist = sorted(glob.glob(input_fnames))

    if not flist:
        print(f"No MAT files found in directory: {input_folder}")
        return

    print(f"No. of files: {len(flist)}")

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Function with curry arguments
    func = poreStructureMaker(
        r_probe=r_probe, len_pixel=len_pixel, save_dir=output_folder
    )

    # Parallel processing
    with mp.Pool(processes=processes) as p:
        p.map(func, flist)

    print("Processing complete.")


if __name__ == "__main__":
    start = time.time()
    prll()  # Click will handle command-line argument parsing
    elpsd = time.time() - start
    print(f"Time to complete: {elpsd:.3f} s")
