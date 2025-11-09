import os, urllib.request

urls = {
    "recyclable": [
        "https://upload.wikimedia.org/wikipedia/commons/8/8d/Plastic_bottle.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/3/33/Soda_can_red.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/4/44/Cardboard_box.jpg"
    ],
    "organic": [
        "https://upload.wikimedia.org/wikipedia/commons/5/52/Banana_peel.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/6/6b/Food_waste.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/a/a5/Apple_core.jpg"
    ]
}

for cls, links in urls.items():
    folder = f"data/train/{cls}"
    os.makedirs(folder, exist_ok=True)
    for i, url in enumerate(links):
        fname = os.path.join(folder, f"{cls}_{i}.jpg")
        print("Downloading:", fname)
        urllib.request.urlretrieve(url, fname)

print("✅ Sample data downloaded!")
