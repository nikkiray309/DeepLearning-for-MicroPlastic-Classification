#https://drive.google.com/drive/folders/1mUYKI0e6F91HSqf0iS4ktQzxHH_Pf_u4?usp=sharing

import gdown

url = "https://drive.google.com/drive/folders/1mUYKI0e6F91HSqf0iS4ktQzxHH_Pf_u4?usp=sharing"
output_path = "./MICRO"  # where you want it locally

gdown.download_folder(url, output=output_path, quiet=False, use_cookies=False)

