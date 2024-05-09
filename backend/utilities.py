import requests, configparser
from urllib.parse import unquote, urlparse
from bs4 import BeautifulSoup


def get_filename_from_cd(cd):
    """
    Get filename from content-disposition
    """
    if not cd:
        return None
    fname = cd.split("filename=")[1]
    if fname.lower().startswith(("utf-8''", "utf-8'")):
        fname = fname.split("'")[-1]
    return unquote(fname)


def get_filename(url):
    filename = urlparse(url).geturl().replace("https://", "").replace("/", "-")
    filename = "content/" + filename
    return filename


def download_file(url):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        filename = get_filename_from_cd(r.headers.get("content-disposition"))
        if not filename:
            filename = urlparse(url).geturl().replace("https://", "").replace("/", "-")
        filename = "content/" + filename
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return filename


def readtext(path):
    path = path.rstrip()
    path = path.replace(" \n", "")
    path = path.replace("%0A", "")

    filename = download_file(path)

    with open(filename, "rb") as f:
        soup = BeautifulSoup(f, "html.parser")
        text = soup.body

    return text


def getconfig():
    config = configparser.ConfigParser()
    config.read("config.ini")
    return dict(config.items("main"))
