from GUI import Grid
import time
import cv2
import imutils
from imutils.perspective import four_point_transform
import numpy as np
from typing import Optional
from imutils import contours
import pytesseract

SUDOKU_BOARD = [['-', '-', '-', '-', '-', '-', '-', '-', '-'],
                ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
                ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
                ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
                ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
                ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
                ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
                ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
                ['-', '-', '-', '-', '-', '-', '-', '-', '-']]



def read_board() -> None:
    """
    uses Opencv library and camera to read a physical Sudoku board. Once the
    empty board matches the actual board, it is the user jobs to quit the
    application since all the numbers have been read
    """
    cap = cv2.VideoCapture("/dev/video0")
    # _, frame = cap.read()
    # cv2.imshow('image', frame)

    while True:
        _, frame = cap.read()
        # frame = take_picture(cap, message) # I have removed the pic feature
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canny = cv2.adaptiveThreshold(gray, 255, \
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                      cv2.THRESH_BINARY_INV, 57, 5)
        kernel = np.ones((3, 3), np.uint8)
        canny = cv2.dilate(canny, kernel, iterations=1)
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # cv2.rectangle(img, (50, 0), (550, 480), (0, 255, 0), 10)
        contours = cv2.findContours(canny.copy(), cv2.RETR_TREE, \
                                    cv2.CHAIN_APPROX_SIMPLE)
        # parses contours to get the appropriate tuple value of the cords #
        contours = imutils.grab_contours(contours)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        peri = cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], 0.01 * peri, True)
        if cv2.contourArea(contours[0]) > 2200 and len(approx) == 4:
            # print('checkpoint2')
            board = four_point_transform(img, approx.reshape(4, 2))
            cv2.imwrite('outbox.png', board)
            res = _extract_and_apply(board)
            if res:
                if _display_board():
                    break
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def _extract_and_apply(board) -> bool:
    """
    Given a coordinates, contour and the image itself, extract the
    number(if exists) out of the box and place it in the the empty Sudoku
    board and update SUDOKU_BOARD
    numpy.ndarray[[[452 231]] [[491 230]] [[491 269]] [[452 269]]]
    """
    # Filter out all numbers and noise to isolate only boxes
    # Load image, grayscale, and adaptive threshold
    image = board
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 57, 5)

    # Filter out all numbers and noise to isolate only boxes
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < 1000:
            cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)
            # totally blacks out the squares

    # Fix horizontal and vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel,
                              iterations=9)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel,
                              iterations=4)

    # Sort by top to bottom and each row by left to right
    invert = 255 - thresh
    # inverts it back to original white and black so we get an empty board
    # but this time all the lines are thick and all squares will be detected

    cnts = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) < 1: return False
    (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

    sudoku_rows = []
    row = []
    count = 0
    for (i, c) in enumerate(cnts, 1):
        area = cv2.contourArea(c)
        # we have found a square
        if area < 50000:
            count += 1
            row.append(c)
            if i % 9 == 0:
                (cnts, _) = contours.sort_contours(row, method="left-to-right")
                sudoku_rows.append(cnts)
                row = []

    if count != 81: return False

    # Iterate through each box
    for i, row in enumerate(sudoku_rows):
        for n, c in enumerate(row):
            mask = np.zeros(image.shape, dtype=np.uint8)
            # fills in a square white on mask
            cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
            # will end up showing the one square on mask
            result = cv2.bitwise_and(image, mask)
            cv2.imwrite('outbox.png', result)
            result[mask == 0] = 255 # make everything on board also white
            number = get_num(c, result)

            SUDOKU_BOARD[i][n] = number if number else ""
            cv2.imshow('result', result)
            cv2.waitKey(1)

    cv2.destroyAllWindows()
    return True


def get_num(contour, img) -> Optional[int]:
    """
    Return the number present in the image otherwise return None
    Note: Adding the config parameter considers the image as raw text
    instead of trying to find paragraphs and line since we are working
    with images with only one number on them
    # """
    x, y, w, h = cv2.boundingRect(contour)
    square = img[y + 3: y + h - 3, x + 3: x + w - 3]
    gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255,\
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                   cv2.THRESH_BINARY, 11, 7)
    ### adding config parameter will take it as raw text ###
    res = pytesseract.image_to_string(thresh, config="--psm 13")
    for ch in res:
        if 49 <= ord(ch) <= 57:
            # print(size)
            return ch


def _display_board() -> bool:
    """
    Using the list representation of the Sudoku board, display the board
    using the class Grid which leverages pygame. This will also give the chance
    for the user to either auto complete the grid or if the grid was read wrong,
    a chance to re-read the grid. Thus return True iff the board was read
    right
    """
    grid = Grid(SUDOKU_BOARD)
    res = grid.create_board()
    print(res)
    if res is False:
        return False
    return True


if __name__ == '__main__':
    """
    Notes: 
    2021/01/05: Detecting 1s and 9s seems to be the only problem
    
    What to do: after the board is read, the user should be able to decide if
    he/she wants to finish the board or try to read the board again since it may
    have read the board wrong(
    """
    start = time.time()
    read_board()
    print(time.time() - start)
    # grid = Grid(CORRECT_BOARD)
    # grid.create_board()
