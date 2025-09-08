import os
import subprocess
import requests
import zipfile
import tarfile
import gdown
import random
import json

# Root data directory
DATA_ROOT = "data"


def makedir(path):
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def download_requests(url, dest, timeout=60):
    """Download using requests, with an added timeout."""
    if os.path.exists(dest):
        print(f"'{os.path.basename(dest)}' already exists, skipping download.")
        return True
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0'
        }
        print(f"Downloading via requests: {url} -> {dest}")
        makedir(os.path.dirname(dest))
        with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"requests download failed for {url}: {e}")
        return False


def download_gdown(file_id, dest):
    """Download from Google Drive using gdown."""
    if os.path.exists(dest):
        print(f"'{os.path.basename(dest)}' already exists, skipping download.")
        return True
    makedir(os.path.dirname(dest))
    try:
        print(f"Downloading from Google Drive (ID: {file_id}) -> {dest}")
        gdown.download(id=file_id, output=dest, quiet=False)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"gdown download failed for {file_id}: {e}")
        return False


def unzip(src, dst):
    """Unzip a ZIP file."""
    if not os.path.exists(src):
        print(f"Zip not found, cannot unzip: {src}")
        return False
    makedir(dst)
    print(f"Unzipping: {src} -> {dst}")
    with zipfile.ZipFile(src, "r") as z:
        z.extractall(dst)
    print("Unzipping complete.")
    return True


def untar(src, dst):
    """Untar a TAR.GZ file."""
    if not os.path.exists(src):
        print(f"TAR not found, cannot untar: {src}")
        return False
    makedir(dst)
    print(f"Untarring: {src} -> {dst}")
    with tarfile.open(src, "r:gz") as t:
        t.extractall(dst)
    print("Untarring complete.")
    return True


def select_top(json_path, top_ratio=0.2):
    """
    Select top samples from a JSON file, compatible with both dict and list formats:
    - If the format is a dict, its values() will be used to generate a list.
    - If elements contain a 'rating' field, select the top_ratio based on descending rating.
    - Otherwise, randomly sample top_ratio of the items.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # If it's a dict, get its values() list
    if isinstance(data, dict):
        items = list(data.values())
    else:
        items = data

    n = len(items)
    top_n = max(1, int(n * top_ratio))

    # Check if 'rating' exists
    first = items[0] if n > 0 else {}
    if isinstance(first, dict) and 'rating' in first:
        sorted_items = sorted(items, key=lambda x: x.get('rating', 0), reverse=True)
        selected = sorted_items[:top_n]
        print(f"Selected top {top_ratio*100:.0f}% ({len(selected)}/{n}) by rating.")
    else:
        selected = random.sample(items, top_n)
        print(f"No 'rating' found; randomly sampled {len(selected)}/{n} items.")
    return selected



if __name__ == "__main__":

    # 1. Single Image Description: COCO Captions 2017
    print("\n--- Setting up COCO Captions 2017 ---")
    coco_dir = os.path.join(DATA_ROOT, "single", "coco")
    makedir(coco_dir)
    # Images
    img_url = "http://images.cocodataset.org/zips/train2017.zip"
    img_zip = os.path.join(coco_dir, "train2017.zip")
    if download_requests(img_url, img_zip):
        unzip(img_zip, coco_dir)
    # Annotations
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ann_zip = os.path.join(coco_dir, "annotations_trainval2017.zip")
    if download_requests(ann_url, ann_zip):
        unzip(ann_zip, coco_dir)

    # 2. Difference Description: Spot-the-Diff
    print("\n--- Setting up Spot-the-Diff (harsh19 version) ---")
    REPO_ZIP_URL = "https://github.com/harsh19/spot-the-diff/archive/refs/heads/master.zip"
    DEST_DIR = os.path.join(DATA_ROOT, "diff", "spot-the-diff-harsh19")
    DEFAULT_UNZIPPED_FOLDER_NAME = "spot-the-diff-master"
    RESIZED_IMAGES_ID = "1OVb4_3Uec_xbyUk90aWC6LFpKsIOtR7v"
    CLUSTER_IMAGES_ID = "1zUxY1CMbNjD64OoK34Z412z7cRfvOHqH"

    if os.path.exists(DEST_DIR):
        print(f"Directory '{DEST_DIR}' already exists. Assuming repo is set up. Skipping.")
    else:
        repo_zip_path = os.path.join(DATA_ROOT, "diff", "repo.zip")
        if download_requests(REPO_ZIP_URL, repo_zip_path):
            unzip(repo_zip_path, os.path.join(DATA_ROOT, "diff"))
            default_unzipped_path = os.path.join(DATA_ROOT, "diff", DEFAULT_UNZIPPED_FOLDER_NAME)
            print(f"Renaming '{default_unzipped_path}' to '{DEST_DIR}'...")
            os.rename(default_unzipped_path, DEST_DIR)
            os.remove(repo_zip_path)

    print("\n--- Handling resized_images ---")
    resized_dir = os.path.join(DEST_DIR, 'resized_images')
    makedir(resized_dir)
    resized_zip = os.path.join(DEST_DIR, 'resized_images.zip')
    if download_gdown(RESIZED_IMAGES_ID, resized_zip):
        unzip(resized_zip, resized_dir)

    print("\n--- Handling cluster_images ---")
    cluster_dir = os.path.join(DEST_DIR, 'cluster_images')
    makedir(cluster_dir)
    if not os.listdir(cluster_dir):
        gdown.download_folder(id=CLUSTER_IMAGES_ID, output=cluster_dir, quiet=False)

    # 3. Relational Description: NLVR²
    print("\n--- Downloading NLVR² (using git clone for Git LFS support) ---")
    rel_dir = os.path.join(DATA_ROOT, "relation", "nlvr2_repo")
    makedir(rel_dir)
    NLVR_REPO_URL = "https://github.com/lil-lab/nlvr.git"
    if not os.listdir(rel_dir):
        try:
            subprocess.run(['git', 'clone', NLVR_REPO_URL, rel_dir], check=True)
            print("NLVR repository cloned successfully.")
        except Exception as e:
            print("Error cloning NLVR². Ensure Git & LFS installed:", e)
    print("NLVR² data files are in:", os.path.join(rel_dir, 'nlvr2', 'data'))



