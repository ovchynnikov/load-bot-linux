# pylint: disable=missing-module-docstring
import os
import shutil
from pathlib import Path
from logger import debug, error


def cleanup(path):
    """
    Cleans up temporary files by deleting the specified video file and its containing directory.

    This function removes the specified path with video or images and
    its parent directory. Logs are printed if debugging is enabled.

    Parameters:
        path (list): The path to the video file to delete or to empty tmp folder.

    Logs:
        Logs messages about the deletion process or any errors encountered.
    """

    if not path:  # Handle empty or None input
        debug("Cleanup func. Path is None or empty. Abort.")
        return
    debug("Cleanup func. path is: %s", path)

    if isinstance(path, list):
        if not path:
            return
        path = path[0]  # Take the first path if list is given.

    path = str(path)  # Ensure path is a string

    p = Path(path)

    if p.is_dir():
        folder_to_delete = path
    else:
        folder_to_delete = str(p.parent)

    debug("Temporary directory to delete %s", folder_to_delete)

    try:
        shutil.rmtree(folder_to_delete)
        if os.path.exists(folder_to_delete):
            error("Temporary directory still exists after cleanup: %s", folder_to_delete)
        else:
            debug("Temporary directory successfully deleted: %s", folder_to_delete)
    except (OSError, IOError) as cleanup_error:
        error("Error deleting folder: %s", cleanup_error)
