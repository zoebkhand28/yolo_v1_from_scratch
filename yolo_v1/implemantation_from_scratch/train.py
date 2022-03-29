import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as ft
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from loss import YoloLoss
from dataset import VOCDataset
from utils.in_out_boxes import *
from utils.nms import non_max_suppression
from utils.mean_avg_precision import mean_average_precision




seed = 123
torch.manual_seed(seed)

# Hyperparameter
learning_rate = 2e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_sizes = 16
weight_decay = 0
epochs = 100
num_workers = 2
pin_memory = True
load_model = False
load_model_file = "overfit.path.tar"
img_dir = r"E:\PascalVOC_dataset\images"
label_dir = r"E:\PascalVOC_dataset\labels"


class Compose(object):
    def __init__(self, trans):
        self.transforms = trans

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(), ])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):

        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")


def main():

    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = YoloLoss()

    # to load a model from a checkpoint
    # if LOAD_MODEL:
    #     load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(
        r"E:\PascalVOC_dataset\100examples.csv",
        transform=transform,
        img_dir=img_dir,
        label_dir=label_dir,
    )

    # test_datasets = VOCDataset(
    #     "E:\\PascalVOC_dataset\\test.csv",
    #     transform=transform,
    #     img_dir=img_dir,
    #     label_dir=label_dir,
    # )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_sizes,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,
    )

    # test_loader = DataLoader(
    #     dataset=test_datasets,
    #     batch_size=batch_sizes,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    #     shuffle=True,
    #     drop_last=True,
    # )

    for epoch in range(epochs):
        print("epoch:", epoch+1)

        train_fn(train_loader, model, optimizer, loss_fn)

        if epoch % 10 == 0:
            pred_boxes = []
            true_boxes = []

            model.eval()
            train_idx = 0

            for batch_idx, (x, label) in enumerate(train_loader):

                if batch_idx > 6:
                    break

                x = x.to(device)
                label = label.to(device)

            with torch.no_grad():
                predictions = model(x)

            batch_size = x.shape[0]
            label_boxes = cellboxes_to_boxes(label)
            bboxes = cellboxes_to_boxes(predictions)

            for idx in range(batch_size):
                nms_boxes = non_max_suppression(
                    bboxes[idx],
                    iou_threshold=0.5,
                    threshold=0.4,
                    box_format="midpoint"
                )

            for nms_box in nms_boxes:
                pred_boxes.append([train_idx]+nms_box)

            for box in label_boxes[idx]:
                if box[1] > 0.4:
                    true_boxes.append([train_idx]+box)

            train_idx += 1

            model.train()

            mean_avg_prec = mean_average_precision(
                pred_boxes, true_boxes, iou_thresholds=0.5, box_format="midpoint"
            )
            print(f"Train mAP:{mean_avg_prec}")


if __name__ == '__main__':

    main()
