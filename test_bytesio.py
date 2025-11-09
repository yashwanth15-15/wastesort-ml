# test_bytesio.py
from io import BytesIO
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

p = r"data\\test\\O\\O_10070.jpg"   # same path as above

try:
    b = open(p, "rb").read()
    buf = BytesIO(b)
    img = Image.open(buf)
    img = img.convert("RGB")
    print("OK — BytesIO load success. Size:", img.size, "Mode:", img.mode)
except Exception as e:
    print("FAILED BytesIO load:", repr(e))
