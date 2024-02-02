import numpy as np


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def is_coordinate_zero(c1, c2, c3):
    if c1 == [0, 0] and c2 == [0, 0] and c3 == [0, 0]:
        return True
    else:
        return False


def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Calculate area of intersection
    area_inter = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    # Calculate area of individual bounding boxes
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate union area
    area_union = area_box1 + area_box2 - area_inter

    # Calculate IoU
    iou = area_inter / area_union if area_union > 0 else 0.0
    return iou


def calculate_all_ious(bounding_boxes):
    num_boxes = len(bounding_boxes)
    ious = []

    for i in range(num_boxes):
        for j in range(i + 1, num_boxes):
            ious.append(calculate_iou(bounding_boxes[i], bounding_boxes[j]))

    return ious


def get_interaction_box(boxes):
    interaction_boxes = []
    for i, iou in enumerate(calculate_all_ious(boxes)):
        if iou > 0.1:
            try:
                # Create interaction box coordinate
                interaction_coordinate = [
                    min(boxes[i][0], boxes[i + 1][0]),  # x1
                    min(boxes[i][1], boxes[i + 1][1]),  # y1
                    max(boxes[i][2], boxes[i + 1][2]),  # x2
                    max(boxes[i][3], boxes[i + 1][3])  # y2
                ]
                interaction_boxes.append(interaction_coordinate)
            except IndexError:
                pass

    return interaction_boxes