# mypy: ignore-errors

import os
import shutil
import urllib.request


# TODO: Unit test.
def download_file(url: str, file_path: str):
    print(f"Downloading file from URL: {url}")
    file_dir, file_name = os.path.split(file_path)
    if file_name == "":
        raise ValueError(f"`file_path` must include a file name, was: {file_path}")
    if not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)  # May raise exception, but that's fine.
    with urllib.request.urlopen(url) as response, open(file_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
