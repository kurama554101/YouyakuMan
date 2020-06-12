import urllib
import os
import gzip
import shutil
import tarfile
import zipfile


def download_and_extract_if_needed(download_url:str, download_path:str, extract_dir_path:str):
    if os.path.exists(download_path):
        print("File {} already exists. Skipped download.".format(download_path))
    else:
        urllib.request.urlretrieve(download_url, download_path)
        print("File downloaded: {}".format(download_path))
        
    if os.path.exists(extract_dir_path):
        print("File {} already exists. Skipped unzip.".format(extract_dir_path))
    else:
        os.makedirs(extract_dir_path, exist_ok=True)
        extract_all(archive_path=download_path, extract_dir_path=extract_dir_path)


def extract_all(archive_path:str, extract_dir_path:str):
    if archive_path.endswith(".tar.gz") or archive_path.endswith("tar"):
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(extract_dir_path)
    elif archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zip:
            zip.extractall(extract_dir_path)
    else:
        raise NotImplementedError("{} file is not able to extract!".format(archive_path))
