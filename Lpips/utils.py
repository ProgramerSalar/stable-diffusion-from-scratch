import torch 
import os, hashlib, requests
from tqdm import tqdm

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}

def md5_has(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()

def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)

    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))

        with tqdm(total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)



def get_ckpt_path(name, root, check=False):

    assert name in URL_MAP, "make sure URL_MAP are Found."
    path = os.path.join(root, CKPT_MAP[name])  # stable-diffusion/vae/LPIPS/vgg.pth

    if not os.path.exists(path) or (check and not md5_has(path) == MD5_MAP[name]):
        print(f"Downloading {name} model from {URL_MAP[name]} to {path}")
        download(URL_MAP[name], path)
        md5 = md5_has(path)
        assert md5 == MD5_MAP[name], md5

    return path 








if __name__ == "__main__":
    flash_hash = md5_has('E:\\Coding-For-YouTube\\Lpips\\example.txt')
    print(flash_hash)

