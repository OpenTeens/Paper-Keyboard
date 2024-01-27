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

    # Use Hough transform to detect circles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=1, maxRadius=20)

    # Initialize list of locating points
    locating_points = []

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
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
                locating_points.append((i[0], i[1]))

    # draw the locating points
    for point in locating_points:
        cv2.circle(img, point, 3, (0, 0, 255), 3)

    # print(len(locating_points))

    if len(locating_points) < 4:
        return [(0, 0), (0, 0), (0, 0), (0, 0)]

    if len(locating_points) > 4:
        locating_points = locating_points[:4]

    # Sort: first by y (100), then by x
    locating_points.sort(key=lambda pos: (pos[1]//500, pos[0]))

    return locating_points


def process(img):
    p = locate(img)
    if not p or len(p) != 4:
        return

    p1, p2, p3, p4 = p

    # Draw the area
    cv2.line(img, p1, p2, (0, 200, 0), 2)
    cv2.line(img, p2, p4, (0, 200, 0), 2)
    cv2.line(img, p4, p3, (0, 200, 0), 2)
    cv2.line(img, p3, p1, (0, 200, 0), 2)
