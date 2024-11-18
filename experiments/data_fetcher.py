import requests
import os
from bs4 import BeautifulSoup


def download_epinions(output_dir="data"):
    """
    Download and extract the Epinions dataset.
    """
    url = "https://snap.stanford.edu/data/soc-Epinions1.txt.gz"
    output_file = os.path.join(output_dir, "soc-Epinions1.txt.gz")

    os.makedirs(output_dir, exist_ok=True)
    response = requests.get(url, verify=False)

    if response.status_code == 200:
        with open(output_file, "wb") as file:
            file.write(response.content)
        print("Epinions dataset downloaded successfully.")
    else:
        print(
            f"Failed to download Epinions dataset. Status code: {response.status_code}"
        )


def download_ciao(output_dir="data"):
    """
    Download the Ciao dataset by scraping the webpage for dataset links.
    """
    url = "https://www.cse.msu.edu/~tangjili/datasetcode/truststudy.htm"
    response = requests.get(url, verify=False)  # SSL verification disabled

    if response.status_code == 200:
        os.makedirs(output_dir, exist_ok=True)
        soup = BeautifulSoup(response.content, "html.parser")
        links = soup.find_all("a")

        dataset_links = [link["href"] for link in links if "dataset" in link["href"]]

        for link in dataset_links:
            if not link.startswith("http"):
                link = f"https://www.cse.msu.edu/~tangjili/datasetcode/{link}"
            file_name = link.split("/")[-1]
            dataset_response = requests.get(link, verify=False)
            if dataset_response.status_code == 200:
                with open(os.path.join(output_dir, file_name), "wb") as file:
                    file.write(dataset_response.content)
                print(f"{file_name} downloaded successfully.")
            else:
                print(
                    f"Failed to download {link}. Status code: {dataset_response.status_code}"
                )
    else:
        print(
            f"Failed to download Ciao dataset page. Status code: {response.status_code}"
        )


if __name__ == "__main__":
    print("Downloading Epinions dataset...")
    download_epinions()

    print("\nDownloading Ciao dataset...")
    download_ciao()

    print("\nDatasets downloaded successfully.")
