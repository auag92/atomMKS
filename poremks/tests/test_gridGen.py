import ase
from toolz.curried import pipe

def test_edtGen():
    import ase.io as aio
    import numpy as np
    import poremks.porosity as pore
    import poremks.grid_generator as gen

    r_Ox = 1.35
    r_Si = 1.35

    len_pixel = 10

    fname = "scripts/MFI.cif"

    cif = fname.split("/")[-1][:-4]
    rep = [1, 1, 1]
    atom = pipe(fname,
                lambda fname: aio.read(fname),
                lambda x: x.repeat(rep))

    S = gen.grid_maker_edt(atom, len_pixel, r_Si, r_Ox, full=False)[0]

    assert S.shape == (202, 198, 133)

    padval = ((1, 1), (1, 1), (0, 0))

    S_dgrid = pipe(S,
                   lambda s: np.pad(S, padval, 'constant', constant_values=0),
                   lambda s: pore.dgrid(S, len_pixel=len_pixel))

    pld  = pore.get_pld(S_dgrid)
    assert np.allclose(pld, 4.4453125, atol=1e-3)

    lcd  = pore.get_lcd(S_dgrid)
    assert np.allclose(lcd, 6.79705810546875, atol=1e-3)

def test_fftGen():
    import ase.io as aio
    import numpy as np
    import poremks.porosity as pore
    from poremks.helpers import sphere
    import poremks.grid_generator as gen

    r_Ox = 1.35
    r_Si = 1.35

    len_pixel = 10

    fname = "scripts/MFI.cif"

    cif = fname.split("/")[-1][:-4]
    rep = [1, 1, 1]
    atom = pipe(fname,
                lambda fname: aio.read(fname),
                lambda x: x.repeat(rep))

    atomSi = sphere(r_Si * len_pixel)
    atomOx = sphere(r_Ox * len_pixel)
    S = gen.grid_maker_fft(atom, len_pixel, r_Si, r_Ox, full=False)[0]

    assert S.shape == (202, 198, 133)

    padval = ((1, 1), (1, 1), (0, 0))

    S_dgrid = pipe(S,
                   lambda s: np.pad(S, padval, 'constant', constant_values=0),
                   lambda s: pore.dgrid(S, len_pixel=len_pixel))

    pld  = pore.get_pld(S_dgrid)
    assert np.allclose(pld, 4.4453125, atol=1e-3)

    lcd  = pore.get_lcd(S_dgrid)
    assert np.allclose(lcd, 6.72309, atol=1e-3)
