import pygame
import pygame.surface
from typing import List, Tuple, Optional
from PIL import Image
import pygame.camera


class Capture(object):
    def __init__(self):
        pygame.init()
        pygame.camera.init()
        self.size = (640,480)
        # create a display surface. standard pygame stuff
        self.display = pygame.display.set_mode(self.size, 0)
        self.cam = pygame.camera.Camera("/dev/video0", (640, 480))

    def main(self):
        self.cam.start()
        going = True
        while going:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    going = False
            image = self.cam.get_image()
            self.display.blit(image, (0, 0))
            pygame.display.flip()


class Grid:
    """
    This is the GUI for the Sudoku solver . the plan is
    that this class is called first to display the board
    and then used for either manually putting numbers into the board
    or solving it automatically

    === Attributes ==
    board: is a 9*9 sudoku board
    """

    def __init__(self, starting_board: List):
        """
        Initialize the Grid
        """
        self.board = starting_board
        self.cells = []

    def create_board(self):
        """
        Creates the starting board interface
        """
        pygame.init()  # initialize pygame
        run = True

        img2 = Image.open(r'images/sudoku_board.jpg')
        img = pygame.image.load(r'images/sudoku_board.jpg')
        display = pygame.display.set_mode((img2.size[0], img2.size[1] + 100))
        pygame.display.set_caption('Sudoku Solver')
        self._create_cells(
            display)  # each cell will be an object that can be changed

        font = pygame.font.Font('freesansbold.ttf', 20)
        font.set_underline(0)
        font.set_bold(0)
        title = font.render('SUDOKU SOLVER', True, (0, 0, 0))

        button_font = pygame.font.Font('freesansbold.ttf', 20)
        button_label = button_font.render('FINISH IT?!', True, (0, 0, 0))
        button2_label = button_font.render('READ AGAIN?!', True, (0, 0, 0))

        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
            if self._read_again():
                return False
            if self._button_pressed():
                res = self.solveSudoku()
                if res:
                    print("the board is completed successfully")
                else:
                    print("an error occurred, read board again")

            display.fill((255, 255, 255))
            display.blit(img, (0, 30))
            display.blit(title, (img2.size[0] // 2 - 90, 10))
            self._update_board()  # the cells are 'blit' onto the display

            pygame.draw.rect(display, (255, 255, 255),
                             pygame.Rect(50, 650, 110, 30))
            display.blit(button_label, (50, 650))
            display.blit(button2_label, (420, 650))
            pygame.display.flip()

    def _read_again(self) -> bool:
        """
        return false iff user presses on the read again button
        """
        pygame.event.get()
        mouse_states = pygame.mouse.get_pressed()
        position = pygame.mouse.get_pos()
        return 420 <= position[0] < 600 and 650 < position[1] < 670 and \
               1 in mouse_states

    def _create_cells(self, display) -> None:
        """
        Creates cells on a sudoku board so that it can be updated
        when we wish
        """
        x, y = 20, 55
        temp = []
        for col1 in range(9):
            temp.append(Cell(x, y, display, self.board[0][col1]))
            x += 65
        self.cells.append(temp)

        for row in range(1, 9):
            x = 20
            y += 65
            temp = []
            for col in range(9):
                temp.append(Cell(x, y, display, self.board[row][col]))
                x += 65
            self.cells.append(temp)

    def _update_board(self) -> None:
        """
        helper function which will fill in the CURRENT board
        NOTE: this needs to be edited since now it it up to date
        """
        for row in range(9):
            for col in range(9):
                self.cells[row][col].blit()

    def _button_pressed(self) -> bool:
        """
        return True iff button was clicked on
        """
        pygame.event.get()
        mouse_states = pygame.mouse.get_pressed()
        position = pygame.mouse.get_pos()
        return 0 < position[0] < 150 and 650 < position[1] < 670 and \
               1 in mouse_states

    def solveSudoku(self) -> bool:
        """
        Do not return anything, modify board in-place instead.
        """
        def backtrack() -> bool:
            x = self._find_empty()
            if not x:
                return True  # base case
            row, col = x
            for number in range(1, 10):
                self.cells[row][col].number = str(number)
                if self._is_valid(number, (row, col)):
                    if backtrack():
                        return True
                    else:
                        self.cells[row][col].number = ''  # empty the cell
                else:
                    self.cells[row][col].number = ''  # empty the cell

            return False
        x = backtrack()
        return x

    def _find_empty(self) -> Optional[Tuple[int, int]]:
        """
        find's any empty cell
        """
        for row in range(9):
            for col in range(9):
                if self.cells[row][col].number == '':
                    return row, col

    def _is_valid(self, number: int, cord: Tuple[int, int]) -> bool:
        """
        Return True iff the number can be placed in the given coordinated
        """
        # check all values in the row
        for col in range(9):
            if col != cord[1] and self.cells[cord[0]][col].number == str(number):
                return False

        # check all values in the col:
        for row in range(9):
            if row != cord[0] and self.cells[row][cord[1]].number == str(number):
                return False

        # check all values in the 3b3 block
        box_x = 3 * (cord[0] // 3)  # gives the row of box
        box_y = 3 * (cord[1] // 3)  # gives the column of box
        for row in range(box_x, box_x + 3):
            for col in range(box_y, box_y + 3):
                if (row, col) != cord and self.cells[row][col].number == self.cells[cord[0]][cord[1]].number:
                    return False

        return True


class Cell:
    """
    Represent the a Cell of the Sudoku board
    """

    def __init__(self, x: int, y: int, display: pygame, number: str):
        self.x = x
        self.y = y
        self.number = number  # either the number of '' for empty cell
        self.img = None
        self.display = display

    def blit(self) -> None:
        """
        blits the given image onto the display/sudoku board
        """
        self.img = pygame.image.load(r'images/basic-numbers/{}.png'.format(self.number)) if self.number != '' else None
        if self.img:
            self.display.fill((255, 255, 255),
                              pygame.Rect(self.x, self.y, 40,
                                          40))
            self.display.blit(self.img, (self.x, self.y))

    def is_empty(self) -> bool:
        """
        Return True iff this cell is empty
        """
        return self.number == ''
