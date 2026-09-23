# import os
# import shutil
# import random
# import xml.etree.ElementTree as ET
# from PIL import Image

# OUT = 'data'
# VAL_SPLIT = 0.2
# random.seed(42)

# # YOLO format wale folders (txt labels)
# YOLO_DIRS = [
#     ('dataset/vid-1', 'vid1'),
#     ('dataset/vid-2', 'vid2'),
#     ('dataset/vid-3', 'vid3'),
# ]

# # Pascal VOC format wale (xml labels) - recursive scan hoga
# VOC_ROOTS = [
#     ('dataset_olx/State-wise_OLX', 'olx'),
#     ('dataset_olx/google_images', 'goog'),
#     ('dataset_olx/video_images', 'vidimg'),
# ]

# IMG_EXT = ('.jpg', '.jpeg', '.png')


# def voc_to_yolo(xml_path, img_path):
#     """XML se YOLO format ki lines banao"""
#     try:
#         root = ET.parse(xml_path).getroot()
#     except Exception:
#         return None

#     # size XML se lo, na mile to image se
#     size = root.find('size')
#     W = H = 0
#     if size is not None:
#         try:
#             W = int(float(size.find('width').text))
#             H = int(float(size.find('height').text))
#         except Exception:
#             pass

#     if W <= 0 or H <= 0:
#         try:
#             W, H = Image.open(img_path).size
#         except Exception:
#             return None

#     lines = []
#     for obj in root.findall('object'):
#         bb = obj.find('bndbox')
#         if bb is None:
#             continue
#         try:
#             x1 = float(bb.find('xmin').text)
#             y1 = float(bb.find('ymin').text)
#             x2 = float(bb.find('xmax').text)
#             y2 = float(bb.find('ymax').text)
#         except Exception:
#             continue

#         if x2 <= x1 or y2 <= y1:
#             continue

#         # clamp - kabhi box image se bahar nikla hota hai
#         x1, x2 = max(0, x1), min(W, x2)
#         y1, y2 = max(0, y1), min(H, y2)

#         cx = ((x1 + x2) / 2) / W
#         cy = ((y1 + y2) / 2) / H
#         w = (x2 - x1) / W
#         h = (y2 - y1) / H

#         if not (0 < w <= 1 and 0 < h <= 1 and 0 <= cx <= 1 and 0 <= cy <= 1):
#             continue

#         lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

#     return lines if lines else None


# pairs = []       # (img_path, label_lines, new_basename)
# stats = {}

# # ---------- YOLO format wale ----------
# for folder, prefix in YOLO_DIRS:
#     if not os.path.isdir(folder):
#         print(f"SKIP - {folder} nahi mila")
#         continue

#     count = 0
#     for f in os.listdir(folder):
#         if not f.lower().endswith(IMG_EXT):
#             continue
#         base = os.path.splitext(f)[0]
#         txt = os.path.join(folder, base + '.txt')
#         if not os.path.exists(txt) or os.path.getsize(txt) == 0:
#             continue

#         with open(txt) as fh:
#             lines = [l.strip() for l in fh if l.strip()]
#         if not lines:
#             continue

#         pairs.append((os.path.join(folder, f), lines, f"{prefix}_{base}"))
#         count += 1
#     stats[prefix] = count

# # ---------- VOC format wale (recursive) ----------
# for root_dir, prefix in VOC_ROOTS:
#     if not os.path.isdir(root_dir):
#         print(f"SKIP - {root_dir} nahi mila")
#         continue

#     count = 0
#     for dirpath, _, files in os.walk(root_dir):
#         for f in files:
#             if not f.lower().endswith(IMG_EXT):
#                 continue
#             base = os.path.splitext(f)[0]
#             xml = os.path.join(dirpath, base + '.xml')
#             if not os.path.exists(xml):
#                 continue

#             img_path = os.path.join(dirpath, f)
#             lines = voc_to_yolo(xml, img_path)
#             if not lines:
#                 continue

#             # state folder ka naam bhi jodo - AP1 aur KA1 na takrayein
#             sub = os.path.basename(dirpath)
#             pairs.append((img_path, lines, f"{prefix}_{sub}_{base}"))
#             count += 1
#     stats[prefix] = count

# if not pairs:
#     raise SystemExit("Kuch nahi mila. Folder paths check karo.")

# # duplicate basename check
# seen = set()
# unique = []
# for p in pairs:
#     if p[2] in seen:
#         continue
#     seen.add(p[2])
#     unique.append(p)
# pairs = unique

# # ---------- split + copy ----------
# if os.path.exists(OUT):
#     shutil.rmtree(OUT)
# for s in ['train', 'val']:
#     os.makedirs(f'{OUT}/{s}/images', exist_ok=True)
#     os.makedirs(f'{OUT}/{s}/labels', exist_ok=True)

# random.shuffle(pairs)
# cut = int(len(pairs) * (1 - VAL_SPLIT))

# for split, items in [('train', pairs[:cut]), ('val', pairs[cut:])]:
#     for img_path, lines, newbase in items:
#         ext = os.path.splitext(img_path)[1]
#         shutil.copy(img_path, f"{OUT}/{split}/images/{newbase}{ext}")
#         with open(f"{OUT}/{split}/labels/{newbase}.txt", 'w') as fh:
#             fh.write('\n'.join(lines) + '\n')

# with open('data.yaml', 'w') as f:
#     f.write(f"""path: {os.path.abspath(OUT)}
# train: train/images
# val: val/images

# nc: 1
# names: ['number_plate']
# """)

# print(f"\n{'='*45}")
# for k, v in stats.items():
#     print(f"{k:<10}: {v}")
# print(f"{'-'*45}")
# print(f"Total     : {len(pairs)}")
# print(f"Train     : {cut}")
# print(f"Val       : {len(pairs)-cut}")
# print(f"{'='*45}")
import os
import shutil
import random
import xml.etree.ElementTree as ET
from PIL import Image

OUT = 'data'
VAL_SPLIT = 0.2
random.seed(42)

YOLO_DIRS = [
    'dataset/vid-1',
    'dataset/vid-2',
    'dataset/vid-3',
]

VOC_ROOTS = [
    'dataset_olx/State-wise_OLX',
    'dataset_olx/google_images',
    'dataset_olx/video_images',
]

IMG_EXT = ('.jpg', '.jpeg', '.png')

bad_label = 0
bad_image = 0


def valid_box(cx, cy, w, h):
    """Coordinates sahi range mein hain?"""
    return (0 < w <= 1 and 0 < h <= 1
            and 0 <= cx <= 1 and 0 <= cy <= 1)


def read_yolo_txt(txt_path):
    """Existing YOLO txt padho aur validate karo"""
    global bad_label
    lines = []
    try:
        with open(txt_path) as fh:
            for raw in fh:
                parts = raw.split()
                if len(parts) != 5:
                    continue
                try:
                    cx, cy, w, h = map(float, parts[1:])
                except ValueError:
                    continue
                if not valid_box(cx, cy, w, h):
                    bad_label += 1
                    continue
                # class hamesha 0 - nc:1 hai
                lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    except Exception:
        return None
    return lines if lines else None


def voc_to_yolo(xml_path, img_path):
    """XML se YOLO format banao"""
    global bad_label
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None

    size = root.find('size')
    W = H = 0
    if size is not None:
        try:
            W = int(float(size.find('width').text))
            H = int(float(size.find('height').text))
        except Exception:
            pass

    # XML mein size na mile ya galat ho to image se lo
    if W <= 0 or H <= 0:
        try:
            W, H = Image.open(img_path).size
        except Exception:
            return None

    lines = []
    for obj in root.findall('object'):
        bb = obj.find('bndbox')
        if bb is None:
            continue
        try:
            x1 = float(bb.find('xmin').text)
            y1 = float(bb.find('ymin').text)
            x2 = float(bb.find('xmax').text)
            y2 = float(bb.find('ymax').text)
        except Exception:
            continue

        if x2 <= x1 or y2 <= y1:
            bad_label += 1
            continue

        # box image ke andar hi rahe
        x1, x2 = max(0.0, x1), min(float(W), x2)
        y1, y2 = max(0.0, y1), min(float(H), y2)
        if x2 <= x1 or y2 <= y1:
            bad_label += 1
            continue

        cx = ((x1 + x2) / 2) / W
        cy = ((y1 + y2) / 2) / H
        w = (x2 - x1) / W
        h = (y2 - y1) / H

        if not valid_box(cx, cy, w, h):
            bad_label += 1
            continue

        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines if lines else None


def image_ok(path):
    """Image khulti hai ya corrupt hai"""
    global bad_image
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        bad_image += 1
        return False


pairs = []      # (img_path, label_lines)

# ---------- YOLO format wale ----------
for folder in YOLO_DIRS:
    if not os.path.isdir(folder):
        print(f"SKIP - {folder} nahi mila")
        continue
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(IMG_EXT):
            continue
        img_path = os.path.join(folder, f)
        txt = os.path.join(folder, os.path.splitext(f)[0] + '.txt')
        if not os.path.exists(txt) or os.path.getsize(txt) == 0:
            continue
        lines = read_yolo_txt(txt)
        if lines and image_ok(img_path):
            pairs.append((img_path, lines))

# ---------- VOC format wale ----------
for root_dir in VOC_ROOTS:
    if not os.path.isdir(root_dir):
        print(f"SKIP - {root_dir} nahi mila")
        continue
    for dirpath, _, files in os.walk(root_dir):
        for f in sorted(files):
            if not f.lower().endswith(IMG_EXT):
                continue
            img_path = os.path.join(dirpath, f)
            xml = os.path.join(dirpath, os.path.splitext(f)[0] + '.xml')
            if not os.path.exists(xml):
                continue
            lines = voc_to_yolo(xml, img_path)
            if lines and image_ok(img_path):
                pairs.append((img_path, lines))

if not pairs:
    raise SystemExit("Kuch nahi mila. Folder paths check karo.")

# ---------- split + copy ----------
if os.path.exists(OUT):
    shutil.rmtree(OUT)
for s in ['train', 'val']:
    os.makedirs(f'{OUT}/{s}/images', exist_ok=True)
    os.makedirs(f'{OUT}/{s}/labels', exist_ok=True)

random.shuffle(pairs)
cut = int(len(pairs) * (1 - VAL_SPLIT))

n = 0
total_boxes = 0
for split, items in [('train', pairs[:cut]), ('val', pairs[cut:])]:
    for img_path, lines in items:
        n += 1
        name = f"{n:05d}"
        ext = os.path.splitext(img_path)[1].lower()
        if ext == '.jpeg':
            ext = '.jpg'

        shutil.copy(img_path, f"{OUT}/{split}/images/{name}{ext}")
        with open(f"{OUT}/{split}/labels/{name}.txt", 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        total_boxes += len(lines)

with open('data.yaml', 'w') as f:
    f.write(f"""path: {os.path.abspath(OUT)}
train: train/images
val: val/images

nc: 1
names: ['number_plate']
""")

print(f"\n{'='*45}")
print(f"Total images  : {len(pairs)}")
print(f"Total boxes   : {total_boxes}")
print(f"Train         : {cut}   (00001 - {cut:05d})")
print(f"Val           : {len(pairs)-cut}   ({cut+1:05d} - {len(pairs):05d})")
print(f"{'-'*45}")
print(f"Galat box hate : {bad_label}")
print(f"Corrupt images : {bad_image}")
print(f"{'='*45}")