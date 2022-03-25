import torch
from iou import intersection_over_union

def non_max_supression(
        bboxes,
        iou_threshold,
        threshold,
        box_format="corners",
):
    assert type(bboxes) == list
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
               or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_formate=box_format
            ) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms
