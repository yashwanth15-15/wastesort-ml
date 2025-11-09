# test_image_check.py
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

p = r"data\\test\\O\\O_10070.jpg"   # change path if that file doesn't exist

try:
    img = Image.open(p)
    img.verify()   # fast check
    print("OK — image verified:", p)
except Exception as e:
    print("FAILED to open:", p)
    print(repr(e))
