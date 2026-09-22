"""FITS structure and placeholder validation for LLAMAS raw frames.

Structure/placeholder companion to the QA engine (``qa_engine.py``). The QA
engine judges pixel statistics per detector; this module answers the structural
questions that come first: are all 24 camera extensions present, does each
extension's header identity match the position the QA config expects, and
which extensions are software placeholders rather than real detector data.

A LLAMAS raw frame is a primary HDU plus 24 image extensions, representing
4 benches x 2 sides x 3 channels (see ``IDX_LOOKUP``). When cameras fail the
acquisition software may drop the extension entirely, or the pipeline may fill
it with a placeholder to keep downstream indexing intact.

Placeholder convention:
    A placeholder is a *constant finite frame at a known placeholder value*:
    finite pixels exist and every one is exactly 1 (real frames) or exactly 0
    (the old pipeline validator, ``create_placeholder_hdu`` here). Any other
    constant frame -- e.g. a railed detector at the ADC ceiling -- is not a
    placeholder and is evaluated normally. Non-finite pixels are ignored, so a
    zero frame that also contains NaNs still counts as a placeholder (the
    pipeline copy, which uses ``np.unique`` over all pixels, says it is not);
    an all-NaN or empty array is never a placeholder. A header ``COMMENT``
    marker written by ``create_placeholder_hdu`` is honoured first so the pixel
    data need not be inspected.

Functions:
    inspect_structure: Header-only structural report used by ``check_image``.
    detector_label: "1.A.Red"-style label from an extension header.
    validate_and_fix_extensions: Check and fix missing extensions (writes placeholders).
    get_expected_camera_list: Complete list of expected camera configurations.
    get_existing_cameras: Camera metadata from existing FITS extensions.
    identify_missing_cameras: Which camera configurations are missing.
    create_placeholder_hdu: Placeholder HDU (zero-valued, pipeline compatibility).
    validate_fits_structure: Checks if a FITS file needs validation.
    get_reference_dimensions: Array dimensions for placeholders.

Example:
    Basic usage to validate and fix a FITS file:

        >>> from llamas_checks.validate import validate_and_fix_extensions
        >>> validate_and_fix_extensions('science.fits')  # fix in place
        >>> validate_and_fix_extensions('science.fits', output_file='science_fixed.fits')
"""

import argparse
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.io import fits

# Set up logger
logger = logging.getLogger(__name__)

# Extension index (1-based HDU number) of every detector in a complete raw frame.
# Keys are (channel, bench, side) with a lowercase channel and the bench as a string.
IDX_LOOKUP: Dict[Tuple[str, str, str], int] = {
    ("red", "1", "A"): 1,
    ("green", "1", "A"): 2,
    ("blue", "1", "A"): 3,
    ("red", "1", "B"): 4,
    ("green", "1", "B"): 5,
    ("blue", "1", "B"): 6,
    ("red", "2", "A"): 7,
    ("green", "2", "A"): 8,
    ("blue", "2", "A"): 9,
    ("red", "2", "B"): 10,
    ("green", "2", "B"): 11,
    ("blue", "2", "B"): 12,
    ("red", "3", "A"): 13,
    ("green", "3", "A"): 14,
    ("blue", "3", "A"): 15,
    ("red", "3", "B"): 16,
    ("green", "3", "B"): 17,
    ("blue", "3", "B"): 18,
    ("red", "4", "A"): 19,
    ("green", "4", "A"): 20,
    ("blue", "4", "A"): 21,
    ("red", "4", "B"): 22,
    ("green", "4", "B"): 23,
    ("blue", "4", "B"): 24,
}
N_DETECTORS = 24

PLACEHOLDER_MARKER = 'Placeholder extension created for missing camera'

# Constant-frame values that mark a placeholder (mirrors qa_engine.PLACEHOLDER_VALUES).
PLACEHOLDER_VALUES = (0.0, 1.0)


def get_expected_camera_list() -> List[Tuple[str, str, str]]:
    """Get the complete list of expected camera configurations.

    Returns:
        List of tuples containing (channel, bench, side) for all 24 expected cameras.

    Example:
        cameras = get_expected_camera_list()
        len(cameras) == 24
        cameras[0] == ('red', '1', 'A')
    """
    return list(IDX_LOOKUP.keys())


def camera_label(camera_config: Tuple[str, str, str]) -> str:
    """Format a (channel, bench, side) tuple as a "1.A.Blue"-style detector label."""
    channel, bench, side = camera_config
    return f"{bench}.{side}.{channel.capitalize()}"


def detector_label(header: fits.Header) -> Optional[str]:
    """Return the "{BENCH}.{SIDE}.{Color}" label for an extension header.

    Uses the BENCH/SIDE/COLOR cards when all three are present, falling back to
    a CAM_NAME card such as ``"1A_red"`` -> ``"1.A.Red"``. Returns ``None`` when
    neither form is available; no positional guessing is ever made.
    """
    try:
        if 'BENCH' in header and 'SIDE' in header and 'COLOR' in header:
            return (f"{str(header['BENCH']).strip()}."
                    f"{str(header['SIDE']).strip().upper()}."
                    f"{str(header['COLOR']).strip().capitalize()}")
        if 'CAM_NAME' in header:
            benchside, color = str(header['CAM_NAME']).strip().split('_', 1)
            return (f"{benchside[0]}.{benchside[1].upper()}."
                    f"{color.strip().capitalize()}")
    except (ValueError, IndexError):
        return None
    return None


def get_existing_cameras(fits_file: str) -> List[Tuple[str, str, str, int]]:
    """Extract camera metadata from existing FITS extensions.

    Detection is header-only: HDUs whose header reports NAXIS < 2 are skipped
    without touching pixel data. This is a deliberate change from the pipeline
    copy, which tested ``hdu.data is None`` (loading and scaling every detector);
    an extension carrying only 1-D data is therefore treated as missing here.

    Args:
        fits_file: Path to the FITS file to analyze.

    Returns:
        List of tuples containing (channel, bench, side, extension_index) for existing cameras.

    Raises:
        FileNotFoundError: If the FITS file does not exist.
        Exception: If there are issues reading the FITS file.
    """
    if not os.path.exists(fits_file):
        raise FileNotFoundError(f"FITS file not found: {fits_file}")

    existing_cameras = []

    try:
        with fits.open(fits_file, memmap=False) as hdul:
            for i, hdu in enumerate(hdul[1:], start=1):  # Skip primary HDU
                if hdu.header.get('NAXIS', 0) < 2:
                    continue

                # Extract camera metadata from header
                try:
                    bench = str(hdu.header['BENCH'])
                    side = hdu.header['SIDE']
                    channel = hdu.header['COLOR'].lower()

                    existing_cameras.append((channel, bench, side, i))

                except KeyError as e:
                    logger.warning(f"Extension {i} missing required header key: {e}")
                    continue

    except Exception as e:
        logger.error(f"Error reading FITS file {fits_file}: {e}")
        raise

    return existing_cameras


def identify_missing_cameras(existing_cameras: List[Tuple[str, str, str, int]],
                             expected_cameras: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """Identify which camera configurations are missing.

    Args:
        existing_cameras: List of existing camera configurations with extension indices.
        expected_cameras: List of all expected camera configurations.

    Returns:
        List of missing camera configurations as (channel, bench, side) tuples.
    """
    # Extract just the camera configuration tuples (ignore extension indices)
    existing_configs = {(channel, bench, side) for channel, bench, side, _ in existing_cameras}
    expected_configs = set(expected_cameras)

    missing = expected_configs - existing_configs

    # Sort missing cameras by their expected index for consistent ordering
    missing_sorted = sorted(missing, key=lambda x: IDX_LOOKUP[x])

    logger.info(f"Found {len(existing_configs)} existing cameras, {len(missing_sorted)} missing")
    if missing_sorted:
        logger.info(f"Missing cameras: {missing_sorted}")

    return missing_sorted


def get_reference_dimensions(fits_file: str) -> Tuple[int, int]:
    """Determine appropriate array dimensions for placeholder extensions.

    Uses the first available extension as a reference for array dimensions.

    Args:
        fits_file: Path to the FITS file.

    Returns:
        Tuple of (height, width) for array dimensions.

    Raises:
        ValueError: If no valid extensions with data are found.
    """
    try:
        with fits.open(fits_file, memmap=False) as hdul:
            for hdu in hdul[1:]:  # Skip primary HDU
                if hdu.data is not None:
                    shape = hdu.data.shape
                    logger.debug(f"Using reference dimensions: {shape}")
                    return shape

        raise ValueError("No extensions with valid data found for reference dimensions")

    except Exception as e:
        logger.error(f"Error determining reference dimensions: {e}")
        raise


def is_placeholder_extension(hdu: fits.ImageHDU) -> bool:
    """Check if an HDU is a placeholder for a missing camera extension.

    Detects placeholder extensions by checking:
    1. Header COMMENT field for the placeholder marker
    2. Data array for a constant finite frame at a placeholder value (finite
       pixels exist, nanmin == nanmax, and that value is 0 or 1)

    Args:
        hdu: FITS ImageHDU to check.

    Returns:
        bool: True if HDU is a placeholder, False otherwise.

    Example:
        >>> with fits.open('science.fits') as hdul:
        ...     if is_placeholder_extension(hdul[5]):
        ...         print("Extension 5 is a placeholder")
    """
    # Check for placeholder marker in header comments
    if 'COMMENT' in hdu.header:
        for comment in hdu.header['COMMENT']:
            if PLACEHOLDER_MARKER in str(comment):
                return True

    # Fallback: a constant finite frame at a known placeholder value (0 or 1)
    if hdu.data is not None:
        try:
            data = np.asarray(hdu.data)
            if data.size == 0 or not np.isfinite(data).any():
                return False
            low, high = float(np.nanmin(data)), float(np.nanmax(data))
            return low == high and low in PLACEHOLDER_VALUES
        except Exception:
            # If we can't analyze the data, assume not a placeholder
            pass

    return False


def get_placeholder_extension_indices(fits_file: str) -> List[int]:
    """Get list of extension indices that are placeholders.

    Scans through all extensions in a FITS file and identifies which ones
    are placeholders for missing camera data.

    Args:
        fits_file: Path to FITS file to scan.

    Returns:
        List of extension indices (1-based) that are placeholders.

    Raises:
        FileNotFoundError: If FITS file does not exist.

    Example:
        >>> placeholder_indices = get_placeholder_extension_indices('science.fits')
        >>> print(f"Found {len(placeholder_indices)} placeholder extensions")
        Found 4 placeholder extensions
    """
    if not os.path.exists(fits_file):
        raise FileNotFoundError(f"FITS file not found: {fits_file}")

    placeholder_indices = []

    try:
        with fits.open(fits_file, memmap=False) as hdul:
            for i in range(1, len(hdul)):  # Skip primary HDU
                if is_placeholder_extension(hdul[i]):
                    placeholder_indices.append(i)

                    # Log camera info if available
                    if logger.isEnabledFor(logging.DEBUG):
                        channel = hdul[i].header.get('COLOR', '?')
                        bench = hdul[i].header.get('BENCH', '?')
                        side = hdul[i].header.get('SIDE', '?')
                        logger.debug(f"Extension {i} ({channel}{bench}{side}) is a placeholder")

        if placeholder_indices:
            logger.info(f"Found {len(placeholder_indices)} placeholder extensions in {os.path.basename(fits_file)}")

    except Exception as e:
        logger.error(f"Error scanning for placeholder extensions: {e}")
        raise

    return placeholder_indices


def create_placeholder_hdu(camera_config: Tuple[str, str, str],
                           reference_shape: Tuple[int, int],
                           extension_name: Optional[str] = None) -> fits.ImageHDU:
    """Create a placeholder HDU with 0.0-valued arrays for a missing camera.

    Zeros are kept for compatibility with the reduction pipeline, which expects
    this convention; the placeholder test accepts all-zeros and all-ones frames.

    Args:
        camera_config: Tuple of (channel, bench, side) for the missing camera.
        reference_shape: Shape tuple (height, width) for the array.
        extension_name: Optional name for the extension.

    Returns:
        FITS ImageHDU with placeholder data and appropriate headers.
    """
    channel, bench, side = camera_config

    # Create array filled with 0.0 values
    # Make these unsigned ints, just like the real data, to avoid type issues in the pipeline
    data = np.zeros(reference_shape, dtype=np.uint16)

    # Create header with camera metadata
    header = fits.Header()
    header['BENCH'] = (bench, 'Bench identifier')
    header['SIDE'] = (side, 'Side identifier (A or B)')
    header['COLOR'] = (channel.upper(), 'Color channel')
    header['EXTNAME'] = (extension_name or f"{channel.upper()}{bench}{side}", 'Extension name')
    header['COMMENT'] = PLACEHOLDER_MARKER
    header['COMMENT'] = 'Data filled with 0.0 values to maintain pipeline compatibility'

    # Create ImageHDU
    hdu = fits.ImageHDU(data=data, header=header)

    logger.debug(f"Created placeholder HDU for {channel}{bench}{side} with shape {reference_shape}")

    return hdu


def validate_fits_structure(fits_file: str, expected_extensions: int = N_DETECTORS) -> Dict[str, Union[bool, int, List]]:
    """Check if a FITS file needs validation and return diagnostic information.

    Args:
        fits_file: Path to the FITS file to check.
        expected_extensions: Number of extensions expected (default: 24).

    Returns:
        Dictionary containing validation results:
        - 'needs_validation': bool indicating if file needs fixing
        - 'current_extensions': int number of current extensions
        - 'missing_count': int number of missing extensions
        - 'existing_cameras': list of existing camera configurations
        - 'missing_cameras': list of missing camera configurations

    Raises:
        FileNotFoundError: If the FITS file does not exist.
    """
    if not os.path.exists(fits_file):
        raise FileNotFoundError(f"FITS file not found: {fits_file}")

    # Get current structure
    try:
        with fits.open(fits_file, memmap=False) as hdul:
            current_extensions = len(hdul) - 1  # Exclude primary HDU
    except Exception as e:
        logger.error(f"Error reading FITS file structure: {e}")
        raise

    # Get camera information
    existing_cameras = get_existing_cameras(fits_file)
    expected_cameras = get_expected_camera_list()
    missing_cameras = identify_missing_cameras(existing_cameras, expected_cameras)

    needs_validation = len(missing_cameras) > 0

    result = {
        'needs_validation': needs_validation,
        'current_extensions': current_extensions,
        'missing_count': len(missing_cameras),
        'existing_cameras': existing_cameras,
        'missing_cameras': missing_cameras
    }

    logger.info(f"Validation check: {current_extensions}/{expected_extensions} extensions present, "
                f"needs_validation={needs_validation}")

    return result


def inspect_structure(fits_path: str, extensions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Header-only structural report for one raw frame.

    Opens the file once (``memmap=False``) and never touches pixel data.

    Args:
        fits_path: Path to the FITS file.
        extensions: Optional list of QA-config extension dicts (``name``,
            ``hdu_index`` and, when available, ``bench``/``side``/``color``).
            Entries lacking any of bench/side/color are skipped for the
            identity comparison.

    Returns:
        Dict with keys:
        - 'n_extensions': image extensions present (``len(hdul) - 1``)
        - 'expected_extensions': ``N_DETECTORS``
        - 'missing_cameras': "1.A.Blue"-style labels of detectors whose HDU is
          absent, in ``IDX_LOOKUP`` order
        - 'identity_mismatches': list of {extension, hdu_index, header_identity}
          for configured extensions whose header identity differs from the
          configured one (skipped when the header carries no identity)
    """
    existing_cameras: List[Tuple[str, str, str, int]] = []
    identity_mismatches: List[Dict[str, Any]] = []

    with fits.open(fits_path, memmap=False) as hdul:
        n_extensions = len(hdul) - 1

        for i, hdu in enumerate(hdul[1:], start=1):
            if hdu.header.get('NAXIS', 0) < 2:
                continue
            # Same header parsing as the identity check (BENCH/SIDE/COLOR or CAM_NAME).
            label = detector_label(hdu.header)
            if label is None:
                continue
            bench, side, color = label.split('.')
            existing_cameras.append((color.lower(), bench, side, i))

        for extension in extensions or []:
            if not all(key in extension for key in ('bench', 'side', 'color')):
                continue
            hdu_index = extension.get('hdu_index')
            if not isinstance(hdu_index, int) or hdu_index >= len(hdul):
                continue
            header_identity = detector_label(hdul[hdu_index].header)
            if header_identity is None:
                continue
            expected = (f"{str(extension['bench']).strip()}."
                        f"{str(extension['side']).strip().upper()}."
                        f"{str(extension['color']).strip().capitalize()}")
            if header_identity != expected:
                identity_mismatches.append({
                    'extension': extension.get('name'),
                    'hdu_index': hdu_index,
                    'header_identity': header_identity,
                })

    missing = identify_missing_cameras(existing_cameras, get_expected_camera_list())

    return {
        'n_extensions': n_extensions,
        'expected_extensions': N_DETECTORS,
        'missing_cameras': [camera_label(config) for config in missing],
        'identity_mismatches': identity_mismatches,
    }


def validate_and_fix_extensions(fits_file: str,
                                expected_extensions: int = N_DETECTORS,
                                output_file: Optional[str] = None,
                                backup: bool = True) -> str:
    """Main function to validate and fix missing extensions in a FITS file.

    This function checks if a FITS file has the expected number of extensions,
    and if not, creates zero-filled placeholder extensions to maintain
    pipeline compatibility.

    Args:
        fits_file: Path to the input FITS file.
        expected_extensions: Number of extensions expected (default: 24).
        output_file: Optional output file path. If None, modifies input file in-place.
        backup: Whether to create a backup of the original file (default: True).

    Returns:
        Path to the output file (either input file or specified output file).

    Raises:
        FileNotFoundError: If the input FITS file does not exist.
        ValueError: If the file has more than expected extensions.
        Exception: For other file I/O or processing errors.

    Example:
        >>> # Fix file in-place with backup
        >>> result_file = validate_and_fix_extensions('science.fits')

        >>> # Create corrected copy
        >>> result_file = validate_and_fix_extensions('science.fits',
        ...                                          output_file='science_fixed.fits')
    """
    if not os.path.exists(fits_file):
        raise FileNotFoundError(f"Input FITS file not found: {fits_file}")

    logger.info(f"Validating FITS file: {fits_file}")

    # Validate current structure
    validation_info = validate_fits_structure(fits_file, expected_extensions)

    if not validation_info['needs_validation']:
        logger.info("File already has correct number of extensions - no changes needed")
        return fits_file

    current_extensions = validation_info['current_extensions']
    missing_cameras = validation_info['missing_cameras']

    if current_extensions > expected_extensions:
        raise ValueError(f"File has more extensions ({current_extensions}) than expected ({expected_extensions})")

    logger.info(f"File needs fixing: {current_extensions}/{expected_extensions} extensions present")
    logger.info(f"Will add {len(missing_cameras)} placeholder extensions")

    # Determine output file
    if output_file is None:
        output_file = fits_file
        if backup:
            backup_file = fits_file + '.backup'
            logger.info(f"Creating backup: {backup_file}")
            shutil.copy2(fits_file, backup_file)

    # Get reference dimensions
    reference_shape = get_reference_dimensions(fits_file)
    logger.info(f"Using reference dimensions: {reference_shape}")

    try:
        # Read original file
        with fits.open(fits_file, memmap=False) as original_hdul:
            # Create new HDU list starting with primary HDU
            new_hdul = fits.HDUList([original_hdul[0].copy()])

            # Track which extensions we've added
            existing_cameras = validation_info['existing_cameras']

            # Create a mapping of expected extension indices to camera configs
            expected_positions = {IDX_LOOKUP[config]: config for config in get_expected_camera_list()}

            # Build the complete HDU list with placeholders for missing cameras
            for ext_idx in range(1, expected_extensions + 1):
                expected_config = expected_positions[ext_idx]

                # Check if this extension exists in the original file
                existing_ext = None
                for channel, bench, side, orig_idx in existing_cameras:
                    if (channel, bench, side) == expected_config:
                        existing_ext = orig_idx
                        break

                if existing_ext is not None:
                    # Copy existing extension
                    new_hdul.append(original_hdul[existing_ext].copy())
                    logger.debug(f"Copied existing extension {existing_ext} for {expected_config}")
                else:
                    # Create placeholder extension
                    placeholder_hdu = create_placeholder_hdu(expected_config, reference_shape)
                    new_hdul.append(placeholder_hdu)
                    logger.info(f"Added placeholder extension for {expected_config} at position {ext_idx}")

            # Write the corrected file
            logger.info(f"Writing corrected file: {output_file}")
            new_hdul.writeto(output_file, overwrite=True)

    except Exception as e:
        logger.error(f"Error processing FITS file: {e}")
        raise

    # Verify the result
    try:
        final_validation = validate_fits_structure(output_file, expected_extensions)
        if final_validation['needs_validation']:
            logger.error("Validation failed after processing - file still needs correction")
            raise RuntimeError("File validation failed after processing")
        else:
            logger.info(f"Successfully validated and fixed FITS file: {output_file}")
            logger.info(f"Final structure: {final_validation['current_extensions']}/{expected_extensions} extensions")

    except Exception as e:
        logger.error(f"Error verifying corrected file: {e}")
        raise

    return output_file


def validate_for_gui(fits_file: str, expected_extensions: int = N_DETECTORS) -> str:
    """Validate and fix a FITS file for GUI extraction, preserving the original.

    Creates a copy with 'GUI_version' in the filename. If extensions need fixing,
    the copy is modified to add placeholders. Original file is never touched.
    If the file already has the expected extensions, returns the original path unchanged.

    Args:
        fits_file: Path to the input FITS file.
        expected_extensions: Number of extensions expected (default: 24).

    Returns:
        Path to use for extraction (original if valid, GUI_version copy if fixed).
    """
    validation_info = validate_fits_structure(fits_file, expected_extensions)

    if not validation_info['needs_validation']:
        logger.info("File already has correct number of extensions - using original")
        return fits_file

    # Build GUI_version output path and copy the original
    base, ext = os.path.splitext(fits_file)
    gui_file = f"{base}_GUI_version{ext}"
    shutil.copy2(fits_file, gui_file)
    logger.info(f"Copied FITS file to: {gui_file}")

    # Fix the copy in-place (modifies gui_file, not original)
    return validate_and_fix_extensions(gui_file, expected_extensions=expected_extensions,
                                       backup=False)


def main() -> int:
    """Command-line interface for FITS validation."""
    parser = argparse.ArgumentParser(description="Validate and fix FITS files with missing camera extensions")
    parser.add_argument("fits_file", help="Input FITS file to validate")
    parser.add_argument("-o", "--output", help="Output file (default: modify input file)")
    parser.add_argument("-e", "--expected", type=int, default=N_DETECTORS,
                        help="Expected number of extensions (default: 24)")
    parser.add_argument("--no-backup", action="store_true",
                        help="Do not create backup when modifying input file")
    parser.add_argument("--check-only", action="store_true",
                        help="Only inspect the structure and print the report as JSON; write nothing")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        if args.check_only:
            print(json.dumps(inspect_structure(args.fits_file), indent=2))
            return 0

        result_file = validate_and_fix_extensions(
            args.fits_file,
            expected_extensions=args.expected,
            output_file=args.output,
            backup=not args.no_backup
        )
        print(f"Successfully processed: {result_file}")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
