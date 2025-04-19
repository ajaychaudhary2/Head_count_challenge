import gdown

url = "https://drive.google.com/drive/folders/1IytaFi_CGeTzBhZugFU_9zfBcBfkY_MS?usp=sharing"
output = "model_final.pth"
gdown.download(url, output, quiet=False)
