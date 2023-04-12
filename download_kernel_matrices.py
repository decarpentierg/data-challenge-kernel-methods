import requests
from tqdm import tqdm

# URL of matrix wlsk.npy
WLSK_URL = "https://drive.google.com/u/0/uc?id=1WXTVWqo2CIzrsMsw7H_2hkXWgGTMZaUR&export=download"
# URL of matrix wlsk_test.npy
WLSK_TEST_URL = "https://drive.google.com/u/0/uc?id=1IbOCzKktn-nsRH16zbzUgq5WK6xoZIWq&export=download"

def download_matrix(url, file_name):
    """Download file from Google Drive.

    Arguments
    ---------
    url: str
        URL of file to download from Gooogle Drive

    file_name: str
        Name of file to save
    """
    # Make a request to the URL
    response = requests.get(url, stream=True)

    # Get the total file size in bytes
    file_size = int(response.headers.get("Content-Length", 0))

    # Create a progress bar
    progress = tqdm(
        response.iter_content(chunk_size=1024), 
        f"Downloading {file_name}", 
        total=file_size, 
        unit="B", 
        unit_scale=True, 
        unit_divisor=1024
    )

    # Write the contents of the response to a file
    with open(file_name, "wb") as f:
        for data in progress.iterable:
            # Write data to file
            f.write(data)
            # Update the progress bar manually
            progress.update(len(data))

def download_wlsk():
    download_matrix(WLSK_URL, "wlsk.npy")

def download_wlsk_test():
    download_matrix(WLSK_TEST_URL, "wlsk_test.npy")
