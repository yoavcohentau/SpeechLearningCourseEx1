import os
import tarfile
import urllib.request

# ==========================
# CONFIG
# ==========================
SAVE_DIR = "J:\My Drive\Courses\YoavAndItayShared\Speech\DataSet_musan"   # change if needed
MUSAN_URL = "https://openslr.org/resources/17/musan.tar.gz"
ARCHIVE_NAME = "musan.tar.gz"

# ==========================
# Create folder
# ==========================
os.makedirs(SAVE_DIR, exist_ok=True)

archive_path = os.path.join(SAVE_DIR, ARCHIVE_NAME)

# ==========================
# Download
# ==========================
if not os.path.exists(archive_path):
    print("Downloading MUSAN dataset...")
    urllib.request.urlretrieve(MUSAN_URL, archive_path)
    print("Download complete.")
else:
    print("Archive already exists.")

# ==========================
# Extract
# ==========================
extract_path = os.path.join(SAVE_DIR, "musan")

if not os.path.exists(extract_path):
    print("Extracting MUSAN...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(SAVE_DIR)
    print("Extraction complete.")
else:
    print("MUSAN already extracted.")

print("MUSAN dataset available at:", extract_path)
