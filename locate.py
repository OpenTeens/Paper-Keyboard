import math
from typing import List
import cv2
import numpy as np


def locate(img: cv2.Mat) -> List[tuple]:
    """
    Locate a area according to four locating points.
    The shape of the locating point: black circle with a white square in the middle.
    :param img: the image to be located
    :return: the coordinate of the rectangle area
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply binary threshold
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize list of ellipses
    ellipses = []

    for contour in contours:
        # Check if the contour has enough points to fit an ellipse
        if len(contour) >= 5:
            # Fit an ellipse and add it to the list
            ellipse = cv2.fitEllipse(contour)
            ellipses.append(ellipse)

    circles = []
    for ellipse in ellipses:
        # Convert to a similar circle
        x, y = ellipse[0]
        r = (ellipse[1][0] + ellipse[1][1]) / 4
        circles.append([x, y, r])

    # Initialize list of locating points
    locating_points = []

    if circles is not None:
        circles = np.around(circles)

        for i in circles:
            # i = [x, y, r]

            # Check if there's a square inside the circle
            has_sq = False

            # Apply binary threshold
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Check if the approximated shape has four points and is approximately square
                if 3 <= len(approx) <= 5:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    if 0.2 <= aspect_ratio <= 5:
                        has_sq = True
                        break

            if has_sq:
                if i[0] > 0 and i[1] > 0:
                    i = int(i[0]), int(i[1])
                    locating_points.append((i[0], i[1]))

    # draw the locating points
    for point in locating_points:
        print(point)
        cv2.circle(img, point, 3, (0, 0, 255), 3)

    if len(locating_points) < 3:
        return [(0, 0), (0, 0), (0, 0), (0, 0)]

    if len(locating_points) == 3:
        # Did not find enough locating points
        # Find two farthest points as a diagonal
        max_dist = 0
        p1 = None
        p2 = None
        for i in range(len(locating_points)):
            for j in range(i+1, len(locating_points)):
                a, b = locating_points[i], locating_points[j]
                dist = math.dist(a, b)

                if dist > max_dist:
                    max_dist = dist
                    p1, p2 = a, b

        # use the 3rd point to determine the 4th point
        p3 = [point for point in locating_points if point not in (p1, p2)][0]

        # mirror p3 to get p4
        pmid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        p4 = (2 * pmid[0] - p3[0], 2 * pmid[1] - p3[1])

        cv2.circle(img, p1, 3, (225, 0, 0), 3)

        locating_points = [p1, p2, p3, p4]

    if len(locating_points) > 4:
        locating_points = locating_points[:4]

    # Sort: first by y (100), then by x
    locating_points.sort(key=lambda pos: (pos[1]//500, pos[0]))

    return locating_points


def process(img):
    p = locate(img)

    p1, p2, p3, p4 = p

    # Draw the area
    cv2.line(img, p1, p2, (0, 200, 0), 2)
    cv2.line(img, p2, p4, (0, 200, 0), 2)
    cv2.line(img, p4, p3, (0, 200, 0), 2)
    cv2.line(img, p3, p1, (0, 200, 0), 2)
