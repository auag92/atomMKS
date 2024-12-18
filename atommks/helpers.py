from pathlib import Path

import numpy as np
import pandas as pd
from toolz.curried import curry, pipe

try:
    import edt

    transform_edt = curry(edt.edt)(black_border=False)
except ImportError:
    import scipy.ndimage.morphology as morph

    transform_edt = morph.distance_transform_edt
    print("you can install edt for speed-up as - ")
    print("pip install edt")
    pass

try:
    import torch

    def imfilter(x_data, f_data, device=None):
        """
        Convolve real-valued f_data over real-valued x_data using RFFT in PyTorch. # noqa: E501
        Supports optional GPU usage for faster computation. # noqa: E501

        Parameters:
        - x_data: torch.Tensor (input data, assumed to be real-valued)
        - f_data: torch.Tensor (filter data, assumed to be real-valued)
        - device: torch.device (optional, to specify computation on 'cuda' or 'cpu') # noqa: E501

        Returns:
        - result: torch.Tensor (convolved result)
        """
        # Determine the device (CPU or GPU)
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        # device = torch.device('cpu')

        # Convert NumPy arrays to PyTorch tensors if needed
        if isinstance(x_data, np.ndarray):
            x_data = torch.tensor(
                x_data, dtype=torch.float64, requires_grad=False
            )
        if isinstance(f_data, np.ndarray):
            f_data = torch.tensor(
                f_data, dtype=torch.float64, requires_grad=False
            )

        # Move the input tensors to the selected device
        x_data = x_data.to(device)
        f_data = f_data.to(device)

        # Perform FFT-based convolution using the pipe pattern
        result = pipe(
            f_data,
            lambda x: torch.fft.ifftshift(
                x
            ),  # Shift the zero frequency component to the center
            lambda x: torch.fft.rfftn(
                x
            ),  # Perform Real FFT on the filter data
            lambda x: torch.conj(x)
            * torch.fft.rfftn(
                x_data
            ),  # Element-wise multiplication with conjugate
            lambda x: torch.fft.irfftn(
                x, s=x_data.shape
            ),  # Perform inverse Real FFT, reshape to original size
            lambda x: torch.abs(x),
        )  # Return the absolute value of the result

        # Move the result back to the CPU for further processing if needed
        return (
            result.cpu().numpy() if device.type == "cuda" else result.numpy()
        )

except ImportError:

    @curry
    def imfilter(x_data, f_data):
        """
        to convolve f_data over x_data
        """
        return pipe(
            f_data,
            lambda x: np.fft.ifftshift(x),
            lambda x: np.fft.fftn(x),
            lambda x: np.conj(x) * np.fft.fftn(x_data),
            lambda x: np.fft.ifftn(x),
            lambda x: np.absolute(x),
        )

    print("you can install torch for speed-up as - ")
    print("conda install pytorch-cpu torchvision-cpu -c pytorch")
    pass


def sphere(r=10):
    """
    args: radius of the sphere

    returns: A 3D cubic matric of dim (2*r+1)^1
    """
    return pipe(
        2 * r + 1,
        lambda x: np.mgrid[:x, :x, :x],
        lambda xx: (xx[0] - r) ** 2 + (xx[1] - r) ** 2 + (xx[2] - r) ** 2,
        lambda x: (x < r * r) * 1,
    )


@curry
def padder(inp, shape, const_val=0):
    """
    args :  input matrix, new shape

    returns : matrix reshaped to given shape
    """
    ls = np.floor((shape - inp.shape) / 2).astype(int)
    hs = np.ceil((shape - inp.shape) / 2).astype(int)
    return np.pad(
        inp,
        ((ls[0], hs[0]), (ls[1], hs[1]), (ls[2], hs[2])),
        "constant",
        constant_values=const_val,
    )


@curry
def return_slice(x_data, cutoff):
    s = (np.asarray(x_data.shape) // 2 + 1).astype(int)
    cutoff = (np.asarray(cutoff) // 2 + 1).astype(int)

    if x_data.ndim == 2:
        return x_data[
            (s[0] - cutoff[0]) : (s[0] + cutoff[0] + 1),
            (s[1] - cutoff[1]) : (s[1] + cutoff[1] + 1),
        ]
    elif x_data.ndim == 3:
        return x_data[
            (s[0] - cutoff[0]) : (s[0] + cutoff[0] + 1),
            (s[1] - cutoff[1]) : (s[1] + cutoff[1] + 1),
            (s[2] - cutoff[2]) : (s[2] + cutoff[2] + 1),
        ]
    else:
        print("Incorrect Number of Dimensions!")


def get_radii(atom_id, radius_type="vdw"):
    """
    atom_id: element symbol
    radius_type = "vdw" for Van der Waals or "cov" for Covalent
    """

    xl = pd.ExcelFile(
        Path(__file__).parents[1] / "assets" / "Elemental_Radii.xlsx"
    )
    df = xl.parse(sheet_name=0, header=2, index_col=1)

    if radius_type == "cov":
        key = 6
    elif radius_type == "vdw":
        key = 7
    else:
        raise ValueError("radius_type not supported")
    if atom_id in df.index:
        return df.loc[atom_id][key]
    else:
        raise ValueError("Elemental symbol not found")


@curry
def write2vtk(matrix, fname="zeo.vtk"):
    """
    args:
    matrix: numpy ndArray
    fname : filename
    """
    sx, sy, sz = matrix.shape
    mx = np.max(matrix)
    # mi = np.min(matrix)
    lines = (
        "# vtk DataFile Version 2.0\nVolume example\nASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS %d %d %d\nASPECT_RATIO 1 1 1\nORIGIN 0 0 0\nPOINT_DATA %d\nSCALARS matlab_scalars float 1\nLOOKUP_TABLE default\n"  # noqa: E501
        % (sx, sy, sz, matrix.size)
    )
    with open(fname, "w") as f:
        f.write(lines)
        for ix in range(sz):
            v = np.ravel(matrix[:, :, ix], order="f")
            v = ["%1.5f" % x for x in np.round(100 * v / mx)]
            line = " ".join(v)
            f.write(line + "\n")
