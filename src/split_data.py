import os, random, shutil

def split_class(src_class_dir, dst_base, val_frac=0.15, test_frac=0.15):
    imgs = [f for f in os.listdir(src_class_dir)
            if os.path.isfile(os.path.join(src_class_dir, f))]
    random.shuffle(imgs)
    n = len(imgs)
    n_val = max(1, int(n * val_frac))
    n_test = max(1, int(n * test_frac))

    cls = os.path.basename(src_class_dir)
    val_imgs = imgs[:n_val]
    test_imgs = imgs[n_val:n_val+n_test]

    for name in val_imgs:
        os.makedirs(os.path.join(dst_base, 'val', cls), exist_ok=True)
        shutil.move(os.path.join(src_class_dir, name),
                    os.path.join(dst_base, 'val', cls, name))
    for name in test_imgs:
        os.makedirs(os.path.join(dst_base, 'test', cls), exist_ok=True)
        shutil.move(os.path.join(src_class_dir, name),
                    os.path.join(dst_base, 'test', cls, name))

    print(f"{cls}: {n - n_val - n_test} train, {n_val} val, {n_test} test")

if __name__ == "__main__":
    base = "data"
    train_dir = os.path.join(base, "train")
    for cls in os.listdir(train_dir):
        path = os.path.join(train_dir, cls)
        if os.path.isdir(path):
            split_class(path, base)
    print("\n✅ Split complete!")
