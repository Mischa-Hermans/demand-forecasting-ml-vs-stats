import os
import gdown
import zipfile

# Output paths
data_dir = "data/m5"
zip_path = os.path.join(data_dir, "m5_raw.zip")

# Make folder
os.makedirs(data_dir, exist_ok=True)

# Google Drive direct file ID (ZIP with full M5 CSVs)
url = "https://drive.google.com/uc?id=1NYHXmgrcXg50zR4CVWPPntHx9vvU5jbM"

# Download file
gdown.download(url, zip_path, quiet=False)

# Extract contents
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(data_dir)

print(f"Files downloaded and extracted to {data_dir}")
