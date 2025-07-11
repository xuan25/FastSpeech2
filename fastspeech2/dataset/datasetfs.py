
import pickle
import tarfile,os
import sys
from typing import IO
import tqdm

class DatasetFS:
    def __init__(self, base_path):
        self.base_path = base_path
        self.tar = tarfile.open(self.base_path)

        self.all_files: list[tarfile.TarInfo] = []
        self.all_folders: list[tarfile.TarInfo] = []

        self.file_map: dict[str, tarfile.TarInfo] = {}

        index_file = f"{self.base_path}.index"
        if os.path.exists(index_file):
            index = pickle.load(open(index_file, "rb"))
            last_modified = index["last_modified"]

            if last_modified == os.path.getmtime(self.base_path):
                print("Index file is up to date, loading from index.")
                self.all_files = index["all_files"]
                self.all_folders = index["all_folders"]
                self.file_map = index["file_map"]
                print(f"{len(self.all_files)} files and {len(self.all_folders)} folders loaded from index.")
                return
            
            print(f"Index file is outdated, re-indexing dataset tarball: {self.base_path}")
        else:
            print(f"Index file not found: {index_file}, generating new index for dataset tarball: {self.base_path}")

        for entry in tqdm.tqdm(self.tar, desc="Indexing dataset tarball", dynamic_ncols=True):
            if entry.isfile():
                self.all_files.append(entry)
                self.file_map[os.path.normpath(entry.name)] = entry
            elif entry.isdir():
                self.all_folders.append(entry)

        print(f"Indexed {len(self.all_files)} files and {len(self.all_folders)} folders in dataset tarball: {self.base_path}")

        last_modified = os.path.getmtime(self.base_path)
        index = {
            "last_modified": last_modified,
            "all_files": self.all_files,
            "all_folders": self.all_folders,
            "file_map": self.file_map
        }
        print(f"Saving index file: {index_file}")
        with open(index_file, "wb") as f:
            pickle.dump(index, f)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if exc_type is not None:
            print(f"Exception occurred: {exc_value}", file=sys.stderr)
        return False

    def close(self):
        if self.tar:
            self.tar.close()
            self.tar = None

    def get_all_files(self):
        return self.all_files
    
    def get_all_folders(self):
        return self.all_folders
    
    def get_file(self, path: str) -> IO[bytes]:
        assert self.tar is not None, "Dataset tarball is not open."

        # search for the file in the tarball
        path_norm = os.path.normpath(path)  # Normalize the path to avoid issues with different path formats
        if path_norm in self.file_map:
            data = self.tar.extractfile(self.file_map[path_norm])
            if data is not None:
                return data
            raise ValueError(f"File {path} is not a valid file in the dataset.")
        
        # if the file is not found, check if it exists in the filesystem
        path_escaped = os.path.join(self.base_path, path_norm)
        if os.path.exists(path_escaped):
            return open(path_escaped, "rb")
        
        # if the file is not found in the tarball or filesystem, raise an error
        raise FileNotFoundError(f"File {path} not found in the dataset.")
    
    def open(self, name: str) -> IO[bytes]:
        """Open a file in the dataset tarball."""
        return self.get_file(name)

