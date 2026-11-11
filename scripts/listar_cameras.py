import cv2
from cv2_enumerate_cameras import enumerate_cameras

cameras = list(enumerate_cameras(cv2.CAP_DSHOW))
print("Cameras encontradas:")
print()
for c in cameras:
    print(f"  [{c.index}] {c.name}")
if not cameras:
    print("  nenhuma")
print()
print("Para o celular (Iriun), use: --camera celular")
