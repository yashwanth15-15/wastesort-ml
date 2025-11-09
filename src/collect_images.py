import cv2, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--classname", required=True)
parser.add_argument("--out", default="data\\train")
parser.add_argument("--count", type=int, default=200)
args = parser.parse_args()

outdir = os.path.join(args.out, args.classname)
os.makedirs(outdir, exist_ok=True)
cap = cv2.VideoCapture(0)

print("Press 'c' to capture, 'q' to quit.")
i = 0
while i < args.count:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("frame", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('c'):
        fname = os.path.join(outdir, f"{args.classname}_{i:04d}.jpg")
        cv2.imwrite(fname, frame)
        i += 1
        print("Saved", fname)
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
