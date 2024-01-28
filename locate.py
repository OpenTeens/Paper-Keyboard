from typing import List, Union
import cv2


def locate(img: cv2.Mat) -> Union[List[tuple], None]:
    """
    Find a paper in an image
    :param img: the image to be located
    :return: the coordinate of the rectangle area
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edged = cv2.Canny(gray, 35, 125)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    points = []
    max_area = 0

    # Find the contour with 4 corners
    for contour in contours:
        # Approximate the contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # If the contour has 4 corners, we have found the paper
        if len(approx) == 4:
            area = cv2.contourArea(approx)

            # If the area is too small, it is probably noise
            if area > max_area:
                max_area = area
                points = approx.reshape(4, 2)

    if max_area > 800:
        return points

    # If the paper is not found, return None
    return None


def process(img):
    p = locate(img)

    if p is None:
        return

    p1, p2, p3, p4 = p

    # Draw the area
    cv2.line(img, p1, p2, (0, 200, 0), 2)
    cv2.line(img, p2, p3, (0, 200, 0), 2)
    cv2.line(img, p3, p4, (0, 200, 0), 2)
    cv2.line(img, p4, p1, (0, 200, 0), 2)
