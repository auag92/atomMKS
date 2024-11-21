import numpy as np
import ase.io as aio
from toolz.curried import pipe
import atommks.porosity as pore
import atommks.grid_generator as gen


def generate_pore_metrics(
    fname,
    radii={"Si": 1.35, "O": 1.35, "H": 1.0},
    len_pixel=10,
    r_probe=0.1, # probe radius
):
    """
    Saves the voxelized structure in matfile format
    """
    try:


        rep=[2, 2, 1]
        padval = ((1, 1), (1, 1), (0, 0))

        S = pipe(
            fname,
            lambda f: aio.read(f).repeat(rep),
            lambda a: gen.grid_maker(a, len_pixel=10, atomic_radii=radii, full=False, fft=True)[0],
            lambda s: np.pad(s, padval, "constant", constant_values=0),
            lambda s: pore.dgrid(s, len_pixel),
        )

        pld = pore.get_pld(S) # calculates the pore limiting diameter, a scaler
        
        lcd = pore.get_lcd(S) # calculates the largest cavity diameter, a scaler

        # generates probe accessible pore region [grid representation] 
        S_1 = (
            pore.gen_cleanPore(
                S, r_probe=r_probe, r_min=2.5, len_pixel=len_pixel
            )
            > 0
        ) * 1

        # generates medial axis of the accessible pore region [grid representation]
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

        # accessible surface area [a scaler]
        asa = pore.get_asa(S_1, len_pixel=10)

        # accessile volume [a scaler]
        av = np.count_nonzero(S_1) * (1 / len_pixel) ** 3

        # pore size distribution [a vector]
        psd = S[S_2 == 1]

        # dimensions of the structure
        dim = np.asarray(S.shape) / len_pixel

        print(fname, pld, lcd, asa, av, n_paths, np.mean(paths), np.mean(psd), dim)

    except Exception as err:
        print("Exception for file : %s" % (fname), err)


if __name__ == "__main__":
    generate_pore_metrics(
        fname="../datasets/iza_zeolites/ABW.cif",
        radii={"Si": 1.35, "O": 1.35, "H": 0.5},
        len_pixel=10,
        r_probe=0.5,
    )