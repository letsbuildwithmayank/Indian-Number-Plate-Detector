from ultralytics import YOLO
import torch


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

    model = YOLO('yolov8m.pt')

    results = model.train(
        data='data.yaml',
        epochs=25,
        imgsz=640,
        batch=8,
        device=0,
        patience=10,
        project='runs',
        name='quick_test',
        exist_ok=True,
        workers=4,

        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    m = results.results_dict
    map50 = m.get('metrics/mAP50(B)', 0)
    map5095 = m.get('metrics/mAP50-95(B)', 0)
    precision = m.get('metrics/precision(B)', 0)
    recall = m.get('metrics/recall(B)', 0)

    print(f"\n{'='*50}")
    print(f"mAP50      : {map50:.3f}   <- sabse important")
    print(f"mAP50-95   : {map5095:.3f}")
    print(f"Precision  : {precision:.3f}   (jo dhoonda wo sach mein plate thi?)")
    print(f"Recall     : {recall:.3f}   (kitni plates dhoondh payi?)")
    print(f"{'='*50}")

    if map50 >= 0.85:
        print("BAHUT BADHIYA - dataset achha hai, bada model chadha do")
    elif map50 >= 0.70:
        print("THEEK HAI - bada model + zyada epochs se aur sudhrega")
    elif map50 >= 0.50:
        print("KAMZOR - chal raha hai par dataset/labels check karne padenge")
    else:
        print("GADBAD - kuch galat hai. Labels ya data.yaml check karo")

    print(f"\nModel: runs/quick_test/weights/best.pt")


if __name__ == '__main__':
    main()