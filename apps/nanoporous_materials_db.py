import asyncio
import concurrent.futures
import logging
import os
from typing import List

import ase.io as aio
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import Column, Float, Integer, String, create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from toolz.curried import pipe

import atommks.grid_generator as gen
import atommks.porosity as pore

logger = logging.getLogger("uvicorn")


class SQLQuery(BaseModel):
    query: str


# FastAPI app initialization
n2dm_db = FastAPI()

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
    slab_height = Column(Float)


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

    pld = pore.get_pld(S)  # calculates the pore limiting diameter, a scaler

    lcd = pore.get_lcd(S)  # calculates the largest cavity diameter, a scaler

    # generates probe accessible pore region [grid representation]
    S_1 = (
        pore.gen_cleanPore(S, r_probe=r_probe, r_min=2.5, len_pixel=len_pixel)
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
        "pld": pld,
        "lcd": lcd,
        "asa": asa,
        "av": av,
        "n_paths": n_paths,
        "avg_path": np.mean(paths),
        "avg_psd": np.mean(psd),
        "slab_height": dim[-1],
    }


# Endpoints


@n2dm_db.post("/add_material/")
async def add_material(
    name: str, file: UploadFile = File(...), db=Depends(get_db)
):
    # Generate a temporary file path
    try:
        filename = os.path.basename(file.filename)
        file_path = f"/tmp/{filename}"

        # Save uploaded file
        with open(file_path, "wb") as f:
            f.write(await file.read())  # Use `await` for async file reading

        # Compute metrics
        try:
            metrics = compute_metrics(
                file_path, {"Si": 1.35, "O": 1.35, "H": 0.5}, 10, 0.5
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error processing CIF: {e}"
            )

        # Save to database
        if db.query(Material).filter(Material.name == name).first():
            raise HTTPException(
                status_code=400,
                detail=f"Material '{name}' already exists in the database.",
            )

        material = Material(name=name, **metrics)
        db.add(material)
        db.commit()

        return {"message": f"Material '{name}' added successfully!"}

    except HTTPException as e:
        db.rollback()
        raise e  # Re-raise HTTP exceptions for proper client feedback
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )
    finally:
        # Ensure temporary file is cleaned up
        if os.path.exists(file_path):
            os.remove(file_path)


# @n2dm_db.post("/populate_db/")
# async def populate_db(files: list[UploadFile] = File(...), db=Depends(get_db)):
#     try:
#         for file in files:
#             # Extract the base file name to avoid path issues
#             filename = os.path.basename(file.filename)

#             # Generate a temporary file path
#             file_path = f"/tmp/{filename}"

#             # Write the uploaded file to the temporary directory
#             with open(file_path, "wb") as f:
#                 f.write(
#                     await file.read()
#                 )  # Use `await` to handle async file reading

#             # Compute metrics and add to DB
#             metrics = compute_metrics(
#                 file_path, {"Si": 1.35, "O": 1.35, "H": 0.5}, 10, 0.5
#             )

#             # Ensure unique material names in the database
#             if db.query(Material).filter(Material.name == filename).first():
#                 raise HTTPException(
#                     status_code=400,
#                     detail=f"Material {filename} already exists in the database.",  # noqa: E501
#                 )

#             material = Material(name=filename, **metrics)
#             db.add(material)

#         db.commit()
#         return {"message": "Database populated successfully!"}

#     except Exception as e:
#         db.rollback()
#         raise HTTPException(
#             status_code=500, detail=f"Error processing files: {e}"
#         )


# Function to process a single file
def process_file(file_data):
    """
    Processes a single file: computes metrics and returns a Material object.
    Args:
        file_data: A tuple containing the file path and filename.
    Returns:
        Material: A Material instance with computed metrics.
    """
    file_path, filename = file_data
    logger.info(f"Processing file: {filename}")
    try:
        metrics = compute_metrics(
            file_path, {"Si": 1.35, "O": 1.35, "H": 0.5}, 10, 0.5
        )
        return Material(name=filename, **metrics)
    except Exception as e:
        logger.error(f"Error processing file {filename}: {e}")
        raise e


# Utility function to run blocking code in an async context
async def run_blocking(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)


# Main endpoint for populating the database
@n2dm_db.post("/populate_db/")
async def populate_db(
    files: List[UploadFile] = File(...),
    db=Depends(get_db),
):
    """
    Endpoint to populate the database with materials from uploaded CIF files.

    Args:
        files (List[UploadFile]): List of uploaded CIF files.
        db (Session): Database session.
        max_workers (int): Maximum number of concurrent processes.

    Returns:
        dict: A success message or an error.
    """
    logger.info("Starting populate_db endpoint")
    file_data_list = []  # To store file paths and filenames

    try:
        # Step 1: Save uploaded files and prepare data for processing
        for file in files:
            filename = os.path.basename(file.filename)
            file_path = f"/tmp/{filename}"
            material_name = filename.split(".")[0]

            # Write the uploaded file to the temporary directory
            with open(file_path, "wb") as f:
                f.write(await file.read())
            logger.info(f"File saved: {file_path}")

            # Ensure unique material names in the database
            if (
                db.query(Material)
                .filter(Material.name == material_name)
                .first()
            ):
                logger.warning(
                    f"Material {material_name} already exists in the database."
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Material {material_name} already exists in the database.",
                )

            file_data_list.append((file_path, material_name))

        # Step 2: Use multiprocessing to process files
        materials = []
        logger.info("Processing files using multiprocessing")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=16
        ) as executor:
            try:
                # Use a timeout for each file processing task
                for material in executor.map(
                    process_file, file_data_list, timeout=60
                ):
                    materials.append(material)
            except concurrent.futures.TimeoutError as e:
                logger.error(f"File processing timeout: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Processing timed out for one or more files.",
                )

        # Step 3: Add all processed materials to the database
        logger.info("Adding processed materials to the database")
        db.add_all(materials)
        db.commit()
        logger.info("Database populated successfully")
        return {"message": "Database populated successfully!"}

    except Exception as e:
        db.rollback()
        logger.error(f"Error processing files: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing files: {e}",
        )


@n2dm_db.get("/query/")
def query_materials(
    name: str = None,
    pld_min: float = None,
    pld_max: float = None,
    db=Depends(get_db),
):
    query = db.query(Material)
    if name:
        query = query.filter(Material.name == name)
    if pld_min is not None:
        query = query.filter(Material.pld >= pld_min)
    if pld_max is not None:
        query = query.filter(Material.pld <= pld_max)
    return query.all()


@n2dm_db.post("/execute_sql/")
async def execute_sql(request: SQLQuery, db=Depends(get_db)):
    """
    Execute an arbitrary SQL query against the database.

    Args:
        query (str): The SQL query to execute.
        db: The database session dependency.

    Returns:
        dict: Result of the query or a success message for non-SELECT queries.
    """
    query = request.query  # Extract the query from the request body
    try:
        # Prepare the SQL statement
        sql = text(query)

        # Check if it's a SELECT query
        if query.strip().lower().startswith("select"):
            # Execute the query and fetch results
            result = db.execute(sql).fetchall()

            # Convert the result into a list of dictionaries
            result_dicts = [
                dict(row._mapping) for row in result
            ]  # Use _mapping for Row compatibility

            return {"result": result_dicts}

        else:
            # Execute non-SELECT queries (e.g., INSERT, UPDATE)
            db.execute(sql)
            db.commit()
            return {"message": "Query executed successfully!"}

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Error executing query: {e}"
        )
