INPUT_TAR = "data/data.tar"
OUTPUT_TAR = "data/LibriTTS.tar"
NEW_ROOT = "data/LibriTTS/"

import tarfile
import tqdm

def name_replace(name: str) -> str|None:
    """Replace the prefix in the tar file names."""
    if name.startswith(NEW_ROOT):
        return name[len(NEW_ROOT):]
    return None

with tarfile.open(INPUT_TAR, "r") as input_tar:
    with tarfile.open(OUTPUT_TAR, "w") as output_tar:
        for member in tqdm.tqdm(input_tar, desc="Processing tar members", dynamic_ncols=True):
            new_name = name_replace(member.name)
            if new_name is None:
                continue
            member_updated = member.replace(name=new_name)
            if member.isfile():
                output_tar.addfile(member_updated, input_tar.extractfile(member))
            else:
                output_tar.addfile(member_updated)
