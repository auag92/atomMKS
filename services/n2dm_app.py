from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import atommks.porosity as pore
import atommks.grid_generator as gen
import ase.io as aio
import numpy as np
import os
from toolz.curried import pipe

# FastAPI app initialization
n2dm_app = FastAPI()

# Database setup

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nano_materials.db")
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# Define the Materials table
class Material(Base):
    __tablename__ = "materials"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    pld = Column(Float)
    lcd = Column(Float)
    asa = Column(Float)
    av = Column(Float)
    n_paths = Column(Integer)
    avg_path = Column(Float)
    avg_psd = Column(Float)
    dim = Column(String)

Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Utility function for metrics computation
def compute_metrics(cif_path, radii, len_pixel, r_probe):
    """
    Computes the pore metrics for a given CIF file.

    Args:
        cif_path (str): Path to the CIF file.
        radii (dict): Atomic radii dictionary.
        len_pixel (int): Pixel length.
        r_probe (float): Probe radius.

    Returns:
        dict: Pore metrics.

    Summary: This function computes the pore metrics for a given CIF file. It voxelizes the structure, calculates the pore limiting diameter, largest cavity diameter, accessible surface area, accessible volume, number of independent transport channels, average path length, average pore size distribution, and dimensions of the structure. # noqa: E501
    """

    rep = [2, 2, 1]
    padval = ((1, 1), (1, 1), (0, 0))

    S = pipe(
        cif_path,
        lambda f: aio.read(f).repeat(rep),
        lambda a: gen.grid_maker(
            a, len_pixel=10, atomic_radii=radii, full=False, fft=True
        )[0],
        lambda s: np.pad(s, padval, "constant", constant_values=0),
        lambda s: pore.dgrid(s, len_pixel),
    )

    pld = pore.get_pld(
        S
    )  # calculates the pore limiting diameter, a scaler

    lcd = pore.get_lcd(
        S
    )  # calculates the largest cavity diameter, a scaler

    # generates probe accessible pore region [grid representation]
    S_1 = (
        pore.gen_cleanPore(
            S, r_probe=r_probe, r_min=2.5, len_pixel=len_pixel
        )
        > 0
    ) * 1

    # generates medial axis of the accessible pore region [grid representation] # noqa: E501
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

    return {
        "pld": pld, "lcd": lcd, "asa": asa, "av": av,
        "n_paths": n_paths, "avg_path": np.mean(paths), "avg_psd": np.mean(psd),
        "dim": "10, 10, 10"
    }

# Endpoints

@n2dm_app.post("/add_material/")
async def add_material(name: str, file: UploadFile = File(...), db=Depends(get_db)):
    # Save uploaded file
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # Compute metrics
    try:
        metrics = compute_metrics(file_path, {"Si": 1.35, "O": 1.35, "H": 0.5}, 10, 0.5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CIF: {e}")

    # Save to database
    material = Material(name=name, **metrics)
    db.add(material)
    db.commit()
    return {"message": f"Material '{name}' added successfully!"}

@n2dm_app.post("/populate_db/")
async def populate_db(files: list[UploadFile] = File(...), db=Depends(get_db)):
    for file in files:
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        # Compute metrics and add to DB
        metrics = compute_metrics(file_path, {"Si": 1.35, "O": 1.35, "H": 0.5}, 10, 0.5)
        material = Material(name=file.filename, **metrics)
        db.add(material)

    db.commit()
    return {"message": "Database populated successfully!"}

@n2dm_app.get("/query/")
def query_materials(name: str = None, pld_min: float = None, pld_max: float = None, db=Depends(get_db)):
    query = db.query(Material)
    if name:
        query = query.filter(Material.name == name)
    if pld_min is not None:
        query = query.filter(Material.pld >= pld_min)
    if pld_max is not None:
        query = query.filter(Material.pld <= pld_max)
    return query.all()
