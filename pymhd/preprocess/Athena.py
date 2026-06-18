# PyMHD: Python for Magnetohydrodynamic Turbulence.
# Copyright (c) 2026 Yuyang Hua (华宇阳)
# License: MIT

"""
pymhd/preprocess/Athena.py
--------------------------

Extract data from Athena++/K/PK output files and build Turbulence object
    - Supports MRI-driven and forced turbulence simulations
    - Supports Athena++ (.athdf), AthenaK (.bin), and AthenaPK (.phdf) output files
    - Supports isothermal and adiabatic EoS
"""

from __future__ import annotations

import importlib.util

import time as timer

from pathlib import Path, PurePath
from typing import Any, Sequence

from .. import ScalarField, VectorField, Turbulence

import yt
yt.set_log_level('ERROR')

# Set print function to flush=True
from functools import partial
print = partial(print, flush=True)

# ==================================================================================================
# Input file parsing
# athinput.* for Athena++, *.athinput for AthenaK, *.in for AthenaPK
# ==================================================================================================

def resolveinputfile(inputfile: str | Path | None) -> Path:
    """Resolve athinput file path from explicit path or glob pattern.

    Parameters
    ----------
    inputfile : str, Path, or None
        Input file path or glob pattern, e.g. 'athinput.hgb'.
        If None, automatically detect default Athena(++/K/PK) input files.

    Returns
    -------
    Path : resolved input file path

    Raises
    ------
    FileNotFoundError : if no valid input file is found
    """
    inputs = [
        "athinput.hgb",
        "athinput.turb",
        "turb.athinput",
        "mri3d_unstratified.athinput",
        "turbulence.in",
    ]

    if inputfile is None:
        matched = [path for name in inputs for path in sorted(Path.cwd().glob(name))]
        if len(matched) != 1:
            raise FileNotFoundError(
                "Cannot uniquely determine input file from known candidates: "
                f"{inputs}. Found {len(matched)} match(es)."
            )
        candidates = matched
    else:
        pattern = str(inputfile)
        if any(c in pattern for c in '*?['):
            if pattern.startswith('./'):
                pattern = pattern[2:]
            candidates = sorted(Path.cwd().glob(pattern))
        else:
            candidates = [Path(inputfile)]

    if not candidates:
        raise FileNotFoundError(f"Cannot find input file matching: {inputfile}")

    return candidates[0]

def parameter(
    inputfile: Path,
    blockname: str,
    params   : Sequence[str],
    defaults : Sequence[Any] | None = None
) -> list[Any]:
    """Extract parameter(s) from a certain block in an athinput file

    Parameters
    ----------
    inputfile : Path, path to the input parameter file
    blockname : str, block name, e.g., "<orbital_advection>"
    params    : Sequence[str], parameter names, e.g., ["Omega0", "qshear"]
    defaults  : Sequence[Any] | None, default values for params not found

    Returns
    -------
    values : list[Any], parameter values in the same order as params

    Raises
    ------
    ValueError        : if parameter not found and no default provided
    FileNotFoundError : if 'inputfile' does not exist
    """
    if not params:
        raise ValueError("At least one parameter should be specified")

    if defaults is not None and len(defaults) != len(params):
        raise ValueError("Number of defaults should match number of params")

    if not inputfile.exists():
        raise FileNotFoundError(f"Input file not found: {inputfile}")

    lines = inputfile.read_text().splitlines()

    # Initialize results dict with defaults
    results: dict[str, Any] = {}
    if defaults is not None:
        results = {p: d for p, d in zip(params, defaults)}

    # Find the blockname section and extract parameters
    inblock = False
    for line in lines:
        # Check if entering target block
        if line.strip() == blockname:
            inblock = True
            continue
        # Check if leaving current block (entering another block)
        if inblock and line.strip().startswith('<') and line.strip().endswith('>'):
            break
        # Look for parameters in current block
        if inblock and '=' in line:
            lhs, rhs = line.split('=', maxsplit=1)
            key = lhs.strip()
            if key in params:
                value = rhs.split('#', maxsplit=1)[0].strip()
                try:
                    results[key] = float(value)
                except ValueError:
                    results[key] = value

    # Check for missing parameters
    for p in params:
        if p not in results:
            raise ValueError(f"Parameter '{p}' not found in {blockname} block of {inputfile}")

    return [results[p] for p in params]


def hasblock(inputfile: Path, blockname: str) -> bool:
    """Check whether an athinput file contains a given block."""
    if not inputfile.exists():
        raise FileNotFoundError(f"Input file not found: {inputfile}")

    lines = inputfile.read_text().splitlines()
    return any(line.strip() == blockname for line in lines)


def AthenaPPinput(inputfile: str | Path) -> dict[str, Any]:
    """Extract parameters from Athena++ input file (athinput.*)

    Currently supports:
        1. Athena++ shearing box simulation: athinput.hgb (type = "MRI")
        2. Athena++ hydrodynamic turbulence simulation: athinput.turb (type = "hydro")

    Parameters
    ----------
    inputfile : str or Path, path to athinput.*

    Returns
    -------
    params : dict, simulation parameters
    """
    inputfile = Path(inputfile)

    x1min, x1max, x2min, x2max, x3min, x3max = parameter(
        inputfile, "<mesh>", ["x1min", "x1max", "x2min", "x2max", "x3min", "x3max"]
    )
    box = (x1max - x1min, x2max - x2min, x3max - x3min)
    Nx, Ny, Nz = parameter(inputfile, "<mesh>", ["nx1", "nx2", "nx3"])
    Omega, q   = parameter(inputfile, "<orbital_advection>", ["Omega0", "qshear"], [0.0, 0.0])
    Cs,        = parameter(inputfile, "<hydro>", ["iso_sound_speed"], [0.0])
    gamma,     = parameter(inputfile, "<hydro>", ["gamma"], [0.0])
    nu, eta    = parameter(inputfile, "<problem>", ["nu_iso", "eta_ohm"], [0.0, 0.0])

    if Cs != 0.0 and gamma != 0.0:
        raise ValueError(
            f"Ambiguous EoS in {inputfile}: "
                "both 'iso_sound_speed' and 'gamma' are defined in <hydro> block."
        )
    if Cs == 0.0 and gamma == 0.0:
        raise ValueError(
            f"Cannot determine EoS in {inputfile}: "
                "neither 'iso_sound_speed' nor 'gamma' is defined in <hydro> block."
        )

    EoS = {
        True : "isothermal",
        False: "adiabatic",
    }[Cs != 0.0]

    params = {
        "box"   : box,
        "Nx"    : int(Nx),
        "Ny"    : int(Ny),
        "Nz"    : int(Nz),
        "Omega" : Omega,
        "q"     : q,
        "Cs"    : Cs,
        "gamma" : gamma,
        "nu"    : nu,
        "eta"   : eta,
        "type"  : "MRI" if (Omega != 0 and q != 0) else "hydro",
        "solver": "FVM",
        "EoS"   : EoS,
    }

    print(f"Code            : Athena++")
    print(f"Turbulence type : {params['type']}")
    print(f"Grid resolution : {Nx} * {Ny} * {Nz}")
    print(f"Box dimensions  : {box[0]} * {box[1]} * {box[2]}")
    print(f"Viscosity       : nu    = {nu}")
    print(f"Resistivity     : eta   = {eta}")
    if EoS == "isothermal":
        print(f"Sound speed     : Cs    = {Cs}")
    else:
        print(f"Adiabatic index : gamma = {gamma}")

    if Omega != 0 or q != 0:
        print(f"Angular velocity: Omega = {Omega}")
        print(f"Shearing rate   : q     = {q}")

    return params


def AthenaKinput(inputfile: str | Path) -> dict[str, Any]:
    """Extract parameters from AthenaK input file (*.athinput).

    Currently supports:
        1. AthenaK shearing box simulation: mri3d_unstratified.athinput (type = "MRI")
        2. AthenaK MHD turbulence simulation: turb.athinput (type = "SSD" or "Bz")
        3. AthenaK hydrodynamic turbulence simulation: turb.athinput (type = "hydro")

    Parameters
    ----------
    inputfile : str or Path, path to *.athinput

    Returns
    -------
    params : dict, simulation parameters
    """
    inputfile = Path(inputfile)

    x1min, x1max, x2min, x2max, x3min, x3max = parameter(
        inputfile, "<mesh>", ["x1min", "x1max", "x2min", "x2max", "x3min", "x3max"]
    )
    Nx, Ny, Nz = parameter(inputfile, "<mesh>", ["nx1", "nx2", "nx3"])
    Omega, q   = parameter(inputfile, "<shearing_box>", ["omega0", "qshear"], [0.0, 0.0])
    ifield,    = parameter(inputfile, "<problem>", ["ifield"], [1])

    hydro = hasblock(inputfile, "<hydro>")
    mhd   = hasblock(inputfile, "<mhd>")
    if hydro == mhd:
        raise ValueError(
            f"Conflicting AthenaK blocks in {inputfile}: "
                "exactly one of <hydro> or <mhd> must be present."
        )

    block = "<hydro>" if hydro else "<mhd>"

    Cs,     = parameter(inputfile, block, ["iso_sound_speed"], [0.0])
    gamma,  = parameter(inputfile, block, ["gamma"], [0.0])
    nu, eta = parameter(inputfile, block, ["viscosity", "ohmic_resistivity"], [0.0, 0.0])

    eos, = parameter(inputfile, block, ["eos"], ["isothermal"])
    eos  = str(eos).lower()

    if eos == "isothermal":
        EoS = "isothermal"
        if Cs == 0.0:
            raise ValueError(
                f"Invalid isothermal EoS in {inputfile}: "
                    f"A non-zero 'iso_sound_speed' must be provided in {block} block."
            )
    elif eos in ["ideal", "adiabatic"]:
        EoS = "adiabatic"
        if gamma == 0.0:
            raise ValueError(
                f"Invalid adiabatic EoS in {inputfile}: "
                    f"A non-zero 'gamma' must be provided in {block} block."
            )
    else:
        raise ValueError(
            f"Unsupported AthenaK EoS '{eos}' in {inputfile}. "
                "Available options: isothermal, ideal."
        )

    if Omega != 0 and q != 0:
        type = "MRI"
    elif hydro:
        type = "hydro"
    elif mhd:
        type = {1: "SSD", 2: "Bz"}[int(ifield)]
    else:
        raise ValueError(f"Unsupported AthenaK turbulence type in {inputfile}. ")

    params = {
        "box"   : (x1max - x1min, x2max - x2min, x3max - x3min),
        "Nx"    : int(Nx),
        "Ny"    : int(Ny),
        "Nz"    : int(Nz),
        "Omega" : Omega,
        "q"     : q,
        "Cs"    : Cs,
        "gamma" : gamma,
        "nu"    : nu,
        "eta"   : eta,
        "type"  : type,
        "solver": "FVM",
        "EoS"   : EoS,
    }

    box = params["box"]
    print(f"Code            : AthenaK")
    print(f"Turbulence type : {params['type']}")
    print(f"Grid resolution : {params['Nx']} * {params['Ny']} * {params['Nz']}")
    print(f"Box dimensions  : {box[0]} * {box[1]} * {box[2]}")
    print(f"Viscosity       : nu    = {params['nu']}")
    print(f"Resistivity     : eta   = {params['eta']}")

    if EoS == "isothermal":
        print(f"Sound speed     : Cs    = {params['Cs']}")
    else:
        print(f"Adiabatic index : gamma = {params['gamma']}")

    if params["Omega"] != 0 or params["q"] != 0:
        print(f"Angular velocity: Omega = {params['Omega']}")
        print(f"Shearing rate   : q     = {params['q']}")

    return params


def AthenaPKinput(inputfile: str | Path) -> dict[str, Any]:
    """Extract parameters from an AthenaPK input file (e.g. turbulence.in).

    Currently supports:
        1. AthenaPK hydrodynamic turbulence simulation: turbulence.in (type = "hydro")
        2. AthenaPK MHD turbulence simulation: turb.in (type = "SSD" or "Bx")

    Parameters
    ----------
    inputfile : str or Path, path to *.in

    Returns
    -------
    params : dict, simulation parameters

    Raises
    ------
    ValueError
    """
    inputfile = Path(inputfile)

    x1min, x1max, x2min, x2max, x3min, x3max = parameter(
        inputfile,
        "<parthenon/mesh>",
        ["x1min", "x1max", "x2min", "x2max", "x3min", "x3max"],
    )
    Nx, Ny, Nz   = parameter(inputfile, "<parthenon/mesh>", ["nx1", "nx2", "nx3"])
    fluid, gamma = parameter(inputfile, "<hydro>", ["fluid", "gamma"], ['glmmhd', 0.0])
    b_config,    = parameter(inputfile, "<problem/turbulence>", ["b_config"], [1])

    if hasblock(inputfile, "<diffusion>"):
        nu, eta = parameter(
            inputfile,
            "<diffusion>",
            ["mom_diff_coeff_code", "ohm_diff_coeff_code"],
            [0.0, 0.0],
        )
    else:
        nu, eta = 0.0, 0.0

    type = {
        "euler" : "hydro",
        "glmmhd": {0: "Bx", 1: "SSD", 2: "SSD"}[int(b_config)],
    }[fluid]

    if gamma == 0.0:
        raise ValueError(f"Missing 'gamma' in {inputfile}: adiabatic EoS requires a non-zero 'gamma'.")

    params = {
        "box"   : (x1max - x1min, x2max - x2min, x3max - x3min),
        "Nx"    : int(Nx),
        "Ny"    : int(Ny),
        "Nz"    : int(Nz),
        "Omega" : 0.0,
        "q"     : 0.0,
        "Cs"    : 0.0,
        "gamma" : gamma,
        "nu"    : nu,
        "eta"   : eta,
        "type"  : type,
        "solver": "FVM",
        "EoS"   : "adiabatic",
    }

    box = params["box"]
    print(f"Code            : AthenaPK")
    print(f"Turbulence type : {params['type']}")
    print(f"Grid resolution : {params['Nx']} * {params['Ny']} * {params['Nz']}")
    print(f"Box dimensions  : {box[0]} * {box[1]} * {box[2]}")
    print(f"Viscosity       : nu    = {params['nu']}")
    print(f"Resistivity     : eta   = {params['eta']}")
    print(f"Adiabatic index : gamma = {params['gamma']}")

    return params


def input2parameters(inputfile: str | Path) -> dict[str, Any]:
    """Extract simulation parameters from an input file

    Dispatches to the appropriate parser based on the input file name.

    Parameters
    ----------
    inputfile : str or Path, path to the input parameter file

    Returns
    -------
    params : dict, extracted parameters with standardized keys:
        - box    : tuple[float, float, float], box dimensions (Lx, Ly, Lz)
        - Nx, Ny, Nz : int, grid resolution
        - Omega  : float, angular velocity
        - q      : float, shear parameter
        - Cs     : float, isothermal sound speed (for isothermal EoS)
        - gamma  : float, adiabatic index (for adiabatic EoS)
        - nu     : float, viscosity
        - eta    : float, resistivity
        - type   : str, turbulence type ("SSD", "Bx", "Bz", "MRI", or "hydro")
        - solver : str, numerical solver ("FVM", "FDM", "SPECTRAL")
        - EoS    : str, equation of state ("isothermal", "adiabatic", "incompressible")

    Raises
    ------
    ValueError : if input file name is not recognized
    """
    inputfile = Path(inputfile)
    filename = inputfile.name

    if PurePath(filename).match("athinput.*"):
        return AthenaPPinput(inputfile)

    if PurePath(filename).match("*.athinput"):
        return AthenaKinput(inputfile)

    if PurePath(filename).match("*.in") and hasblock(inputfile, "<parthenon/mesh>"):
        return AthenaPKinput(inputfile)

    raise ValueError(
        f"Unrecognized input file: '{filename}'. "
        "Supported: athinput.* (Athena++), *.athinput (AthenaK), or *.in (AthenaPK)."
    )


# ==================================================================================================
# Output data extraction
# .athdf for Athena++, .bin for AthenaK, .phdf for AthenaPK
# For AthenaK, .bin files need to be converted to .athdf files first using
#     helper/bin_convert.py, make_athdf.py
# ==================================================================================================

def resolveoutputs(outputs: str | Path | Sequence[str | Path]) -> list[Path]:
    """Resolve output file paths from a list or glob pattern

    Parameters
    ----------
    outputs : str, Path, or list of str/Path
        If str containing glob characters (*, ?, [), treated as a glob pattern.
        Otherwise treated as explicit file path(s).

    Returns
    -------
    list[Path] : sorted list of resolved output file paths

    Raises
    ------
    FileNotFoundError : if no output files match the pattern
    """
    if isinstance(outputs, (str, Path)):
        pattern = str(outputs)
        if any(c in pattern for c in '*?['):
            # Glob pattern: strip leading ./ if present
            if pattern.startswith('./'):
                pattern = pattern[2:]
            resolved = sorted(Path.cwd().glob(pattern))
        else:
            filepath = Path(pattern)
            if not filepath.exists():
                raise FileNotFoundError(f"Output file not found: {filepath}")
            resolved = [filepath]
    else:
        # List of file paths
        resolved = sorted(Path(output) for output in outputs)

    if not resolved:
        raise FileNotFoundError(f"No output files found matching: {outputs}")

    return resolved


def bin2athdf(binfiles: Sequence[Path]) -> list[Path]:
    """Convert AthenaK .bin files to Athena++ .athdf files.

    Uses the bin_convert.py module from AthenaK.

    Parameters
    ----------
    binfiles : Sequence[Path], paths to AthenaK .bin files

    Returns
    -------
    list[Path] : converted Athena++ .athdf paths
    """
    if not binfiles:
        raise FileNotFoundError("No .bin files provided for conversion.")

    path = Path(__file__).resolve().parent / "helper" / "bin_convert.py"
    spec = importlib.util.spec_from_file_location("athenak_bin_convert", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load bin_convert module: {path}")

    bin_convert = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bin_convert)

    converted: list[Path] = []
    for binfile in sorted(binfiles):
        if binfile.suffix != ".bin":
            raise ValueError(f"Unsupported file suffix for conversion: {binfile}")

        anchor = None
        for parent in binfile.parents:
            if parent.name == "bin":
                anchor = parent.parent
                break
        if anchor is None:
            raise ValueError(f"Cannot locate '<root>/bin' anchor for file: {binfile}")

        athdf = anchor / "athdf"
        athdf.mkdir(parents=True, exist_ok=True)
        file = athdf / f"{binfile.stem}.athdf"

        # AthenaK parallel outputs are usually split into rank_* directories.
        # Read from rank_00000000 and gather all ranks in one .athdf output.
        if binfile.parent.name.startswith("rank_"):
            if binfile.parent.name != "rank_00000000":
                continue
            filedata = bin_convert.read_all_ranks_binary(str(binfile))
        else:
            filedata = bin_convert.read_binary(str(binfile))

        bin_convert.write_athdf(str(file), filedata)
        converted.append(file)

    if not converted:
        raise FileNotFoundError(
            "No convertible .bin files found. "
            "For rank-split outputs, make sure rank_00000000 files are included."
        )

    return converted


def output2turbulence(
    outputs  : str | Path | Sequence[str | Path] | None = None,
    outn     : str | None = None,
    inputfile: str | Path | None = None,
    t1       : float | None = None,
    t2       : float | None = None,
    casename : str   | None = None,
) -> Turbulence:
    """Extract data from Athena output files and build a Turbulence object

    Parameters
    ----------
    outputs   : str, Path, Sequence[str | Path], or None; output file paths or glob pattern string
                    e.g. './outputs/*.prim.*.athdf' (Athena++) or './outputs/*.prim.*.phdf' (AthenaPK)
                    If None, a default pattern inferred from input filename is adopted.
    outn      : str | None, output number, defaults to None
    inputfile : str or Path; path or glob pattern of input parameter file, e.g. 'athinput.hgb'
                    If None, automatically detect default Athena(++/K/PK) input files.
    t1        : float | None, start time for filtering, defaults to None (no lower limit)
    t2        : float | None, end time for filtering, defaults to None (no upper limit)
    casename  : str   | None, case name, defaults to None (use parent directory name)

    Returns
    -------
    Turbulence: Turbulence object containing extracted field data

    Raises
    ------
    FileNotFoundError: if no matching files found
    ValueError       : if no valid data found in the specified time range
    """
    start_time = timer.time()
    inputfile = resolveinputfile(inputfile)

    print("")
    print("┌──────────────────────────────────────┐")
    print("│                                      │")
    print("│       PyMHD: Data Preprocessing      │")
    print("│                                      │")
    print("└──────────────────────────────────────┘")

    print("\n═════════ Parameter Extraction ═════════\n")

    params = input2parameters(inputfile)
    box = params["box"]

    print("\n════════════ Data Extraction ═══════════\n")

    if outputs is not None and outn is not None:
        raise ValueError("'outputs' and 'outn' cannot be specified at the same time.")

    defaultoutputs = {
        "athinput.hgb"                : "./*/*.prim.*.athdf",
        "athinput.turb"               : "./*/*.prim.*.athdf",
        "turb.athinput"               : "./*/bin/*.prim.*.bin",
        "mri3d_unstratified.athinput" : "./*/bin/*.prim.*.bin",
        "turbulence.in"               : "./*/parthenon.prim.*.phdf",
    }

    outn2outputs = {
        "athinput.hgb"                : f"./*/*.{outn}.*.athdf",
        "athinput.turb"               : f"./*/*.{outn}.*.athdf",
        "turb.athinput"               : f"./*/bin/*.{outn}.*.bin",
        "mri3d_unstratified.athinput" : f"./*/bin/*.{outn}.*.bin",
        "turbulence.in"               : f"./*/parthenon.{outn}.*.phdf",
    }

    if outputs is None:
        if inputfile.name not in outn2outputs:
            raise ValueError(f"Cannot infer outputs for input file: {inputfile.name}")

        if outn is None:
            outputs = defaultoutputs[inputfile.name]
        else:
            outputs = outn2outputs[inputfile.name]

    outputfiles = resolveoutputs(outputs)

    # Validate suffixes and convert all .bin files to .athdf when present.
    allowed  = {".bin", ".athdf", ".phdf"}
    suffixes = {path.suffix.lower() for path in outputfiles}
    unsupported = sorted(suffixes - allowed)
    if unsupported:
        raise ValueError(f"Unsupported file(s): {', '.join(unsupported)}. Supported: .athdf, .bin, .phdf")

    binfiles   = [path for path in outputfiles if path.suffix.lower() == ".bin"]
    athdffiles = [path for path in outputfiles if path.suffix.lower() == ".athdf"]
    phdffiles  = [path for path in outputfiles if path.suffix.lower() == ".phdf"]

    if binfiles:
        print("AthenaK .bin file(s) detected, converting to .athdf under outputs/athdf/ ...")
        converted = bin2athdf(binfiles)
        outputfiles = sorted({*athdffiles, *phdffiles, *converted})
    else:
        outputfiles = sorted({*athdffiles, *phdffiles})

    # Extract field data from .athdf files using yt
    rhos : list[ScalarField]        = []
    ps   : list[ScalarField | None] = []
    Vs   : list[VectorField]        = []
    Bs   : list[VectorField | None] = []
    accs : list[VectorField | None] = []
    times: list[float]              = []

    def getFields(cg, ds, names: Sequence[str], strict: bool = False):
        """Read the first existing Athena(++/K/PK) field from candidate names

        yt field types: 'athena_pp' (Athena++/K), 'parthenon' (AthenaPK).
        When strict is False, return None if no candidate exists.
        """
        for name in names:
            for ftype in ("athena_pp", "parthenon"):
                key = (ftype, name)
                if key in ds.field_list:
                    return cg[key].value
        if strict:
            raise KeyError(f"Cannot find 'athena_pp' or 'parthenon' field in candidates: {names}")

        return None

    # Step 1: collect all files and times in [t1, t2]
    filetime: list[tuple[Path, float]] = []
    for file in outputfiles:
        try:
            ds = yt.load(file)
            time = float(ds.current_time)
            if (t1 is None or t1 <= time) and (t2 is None or time <= t2):
                filetime.append((file, time))
        except Exception as e:
            print(f"Warning: Error reading file {file}: {e}")
            continue

    if not filetime:
        T1 = str(t1) if t1 is not None else '0'
        T2 = str(t2) if t2 is not None else '∞'
        raise ValueError(f"No valid data found in time range [{T1}, {T2}]")

    # De-duplicate and sort times, where set ({... }) is the set of times
    times = sorted(set({time for _, time in filetime}))

    # Build a map: time -> all files at this time (preserve original file order)
    time2files: dict[float, list[Path]] = {time: [] for time in times}
    for file, time in filetime:
        time2files[time].append(file)

    # Step 2: for each unique time, try extracting all target fields from all files at this time
    for time in times:
        rho: ScalarField | None = None
        p  : ScalarField | None = None
        V  : VectorField | None = None
        B  : VectorField | None = None
        acc: VectorField | None = None
        requireB = params["type"] != "hydro"

        for file in time2files[time]:
            try:
                ds = yt.load(file)
                cg = ds.covering_grid(
                    level=0,
                    left_edge=ds.domain_left_edge,
                    dims=ds.domain_dimensions
                )

                if rho is None or V is None:
                    rhodata = getFields(cg, ds, ["rho", "dens", "prim_density"])
                    Vx = getFields(cg, ds, ["vel1", "velx", "prim_velocity_1"])
                    Vy = getFields(cg, ds, ["vel2", "vely", "prim_velocity_2"])
                    Vz = getFields(cg, ds, ["vel3", "velz", "prim_velocity_3"])

                    if all(field is not None for field in (rhodata, Vx, Vy, Vz)):
                        assert rhodata is not None
                        assert Vx is not None and Vy is not None and Vz is not None
                        rho = ScalarField(rhodata, box)
                        V   = VectorField(Vx, Vy, Vz, box)

                if requireB and B is None:
                    Bx = getFields(cg, ds, ["Bcc1", "bcc1", "prim_magnetic_field_1"])
                    By = getFields(cg, ds, ["Bcc2", "bcc2", "prim_magnetic_field_2"])
                    Bz = getFields(cg, ds, ["Bcc3", "bcc3", "prim_magnetic_field_3"])
                    if all(field is not None for field in (Bx, By, Bz)):
                        assert Bx is not None and By is not None and Bz is not None
                        B = VectorField(Bx, By, Bz, box)

                if params["EoS"] == "adiabatic" and p is None:
                    pdata = getFields(cg, ds, ["press", "p", "prim_pressure"])
                    if pdata is not None:
                        p = ScalarField(pdata, box)
                    else:
                        # AthenaK (non-GR) stores internal energy density "eint" for "mhd_w_bcc" outputs
                        # instead of pressure; convert via P = (gamma-1) * eint
                        eintdata = getFields(cg, ds, ["eint"])
                        if eintdata is not None:
                            assert params["gamma"] is not None
                            p = ScalarField(eintdata * (params["gamma"] - 1.0), box)

                if params["type"] != "MRI" and acc is None:
                    accx = getFields(cg, ds, ["force1", "acc1", "accx", "acc_0"])
                    accy = getFields(cg, ds, ["force2", "acc2", "accy", "acc_1"])
                    accz = getFields(cg, ds, ["force3", "acc3", "accz", "acc_2"])
                    if all(field is not None for field in (accx, accy, accz)):
                        assert accx is not None and accy is not None and accz is not None
                        acc = VectorField(accx, accy, accz, box)

                # Check if all required fields are present
                if (
                    rho is not None and V is not None and (not requireB or B is not None)
                    and (params["EoS"] != "adiabatic" or p is not None)
                ):
                    break

            except Exception as e:
                print(f"Error reading file {file}: {e}")
                continue

        # Required fields at each time: rho/V, B only for MHD, p for adiabatic EoS
        if rho is None or V is None or (requireB and B is None):
            raise ValueError(
                f"Missing required fields (rho, V, or B) at time={time}. "
                f"Files at this time: {[str(path) for path in time2files[time]]}"
            )
        if params["EoS"] == "adiabatic" and p is None:
            raise ValueError(
                f"Missing required pressure field at time={time} for adiabatic EoS. "
                f"Files at this time: {[str(path) for path in time2files[time]]}"
            )

        rhos.append(rho)
        Vs.append(V)
        Bs.append(B)
        accs.append(acc)

        if params["EoS"] == "adiabatic":
            ps.append(p)

    if not times:
        T1 = str(t1) if t1 is not None else '0'
        T2 = str(t2) if t2 is not None else '∞'
        raise ValueError(f"No valid data found in time range [{T1}, {T2}]")

    print(
        f"{len(filetime)} .athdf/.phdf file(s) found in range, "
        f"{len(times)} unique snapshot(s), time range: [{min(times)}, {max(times)}]\n"
    )

    # Default case name: the parent directory of the input file
    if casename is None:
        casename = inputfile.resolve().parent.name

    turbulence = Turbulence(
        case   = casename,
        type   = params["type"],
        solver = params["solver"],
        rhos   = rhos,
        ps     = ps if params["EoS"] == "adiabatic" else [],
        Vs     = Vs,
        Bs     = Bs,
        accs   = accs,
        times  = times,
        Omega  = params["Omega"],
        q      = params["q"],
        EoS    = params["EoS"],
        Cs     = params["Cs"] if params["EoS"] == "isothermal" else None,
        gamma  = params["gamma"] if params["EoS"] == "adiabatic" else None,
        nu     = params["nu"],
        eta    = params["eta"]
    )

    end_time = timer.time()
    print(f"Turbulence object constructed! Total time: {end_time - start_time:.2f} s\n")

    return turbulence
