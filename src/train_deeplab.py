"""Train DeepLabv3+ on the Krestenitis SAR oil-spill dataset (request access:
m4d.iti.gr/oil-spill-detection-dataset/). NOT run in the demo; the trained
checkpoint replaces detect.classify()/dark_blobs() behind the same interface.
Usage: pip install torch segmentation-models-pytorch opencv-python
       python src/train_deeplab.py data/ds
Report PER-CLASS IoU (esp. class 1 oil-spill), never overall accuracy.
"""
import sys, glob, os
import numpy as np, cv2, torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

CLASSES, SIZE = 5, (640, 320)  # 0 sea,1 oil,2 look-alike,3 ship,4 land

class OilSet(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.mask_dir = mask_dir
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        p = self.imgs[i]
        img = cv2.resize(cv2.imread(p, 0), SIZE)
        m = cv2.resize(cv2.imread(os.path.join(
            self.mask_dir, os.path.basename(p).replace(".jpg", ".png")), 0),
            SIZE, interpolation=cv2.INTER_NEAREST)
        return (torch.from_numpy(np.repeat(img[None], 3, 0)).float() / 255.0,
                torch.from_numpy(m).long())

def per_class_iou(pred, y, ncls=CLASSES):
    ious = []
    for c in range(ncls):
        i = ((pred == c) & (y == c)).sum().item()
        u = ((pred == c) | (y == c)).sum().item()
        ious.append(i / u if u else float("nan"))
    return ious

def main(root):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.DeepLabV3Plus("resnet34", encoder_weights="imagenet",
                              in_channels=3, classes=CLASSES).to(dev)
    tr = DataLoader(OilSet(f"{root}/train/images", f"{root}/train/labels"),
                    batch_size=4, shuffle=True, num_workers=2)
    te = DataLoader(OilSet(f"{root}/test/images", f"{root}/test/labels"),
                    batch_size=4)
    w = torch.tensor([0.2, 3.0, 2.0, 3.0, 0.5], device=dev)  # sea dominates
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for ep in range(40):
        model.train(); tot = 0.0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = lossf(model(x), y)
            loss.backward(); opt.step(); tot += loss.item()
        model.eval(); ious = []
        with torch.no_grad():
            for x, y in te:
                p = model(x.to(dev)).argmax(1).cpu()
                ious.append(per_class_iou(p, y))
        miou = np.nanmean(np.array(ious), 0)
        print(f"ep {ep} loss {tot/len(tr):.3f} IoU sea/oil/look/ship/land "
              + "/".join(f"{v:.2f}" for v in miou))
        torch.save(model.state_dict(), "models/deeplab_oil.pt")

if __name__ == "__main__":
    main(sys.argv[1])
