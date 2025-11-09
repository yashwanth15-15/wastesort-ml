import os, urllib.request

os.makedirs("data/train/recyclable", exist_ok=True)
os.makedirs("data/train/organic", exist_ok=True)

recyclable_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/2/24/Plastic_bottles_for_recycling.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/5/56/Aluminium_cans.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/8/83/Glass_bottles_in_a_bin.jpg",
]

organic_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/7/7b/Food_waste_in_bin.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/d/d1/Compost_bin_contents.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/e/e1/Fruit_peels.jpg",
]

for i, url in enumerate(recyclable_urls):
    fname = f"data/train/recyclable/recyclable_{i}.jpg"
    print("Downloading:", fname)
    urllib.request.urlretrieve(url, fname)

for i, url in enumerate(organic_urls):
    fname = f"data/train/organic/organic_{i}.jpg"
    print("Downloading:", fname)
    urllib.request.urlretrieve(url, fname)

print("✅ Download complete!")
