import torch


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculate intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding boxes (BATCH_SIZE, 4)
        box_formate (str): midpoint / corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
    tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":

        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x1 - box1_x2) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x1 - box2_x2) * (box2_y1 - box2_y2))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


if __name__ == '__main__':
    example_input = torch.tensor([[-3, -2, -1], [0, 1, 2]])
    print(type(example_input))