import pygame
import numpy as np
from quarto_env.constants import BOARD_SIZE, PIECE_FEATURES

class Renderer:
    def __init__(self):
        """
        Initialize the Renderer with a Pygame screen and set up the display.
        """
        self.window_size = 600
        self.cell_size = self.window_size // BOARD_SIZE
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption('Quarto')
        self.clock = pygame.time.Clock()

    def render(self, board):
        """
        Render the current state of the board.

        Parameters:
        - board (np.ndarray): The current board state.


        Returns:
        - np.ndarray: The rendered image as a 3D numpy array.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        self.screen.fill((255, 255, 255))
        self.draw_grid()
        self.draw_pieces(board) 
        pygame.display.flip()
        self.clock.tick(30)
        return np.array(pygame.surfarray.array3d(self.screen))

    def draw_grid(self):
        """
        Draw the grid lines on the board.
        """
        for x in range(0, self.window_size + 1, self.cell_size):
            pygame.draw.line(self.screen, (0, 0, 0), (x, 0), (x, self.window_size))
        for y in range(0, self.window_size + 1, self.cell_size):
            pygame.draw.line(self.screen, (0, 0, 0), (0, y), (self.window_size, y))

    def draw_pieces(self, board): 
        """
        Draw the pieces on the board according to the board state.

        Parameters:
        - board (np.ndarray): The current board state.
        """
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                piece_id = int(board[x, y])
                if piece_id == -1: # empty
                    continue
                if piece_id not in PIECE_FEATURES:
                    continue

                # Quarto board stores piece IDs (0..15)
                # Use piece features to choose appearance
                color_bit, height_bit, shape_bit, fill_bit = PIECE_FEATURES[piece_id]
                color = (40, 40, 40) if color_bit == 0 else (220, 220, 220)
                piece_margin = 18
                px = y * self.cell_size + piece_margin
                py = x * self.cell_size + piece_margin
                psize = self.cell_size - (2 * piece_margin)

                if height_bit == 1:
                    psize = int(psize * 0.75)
                    px += int((self.cell_size - 2 * piece_margin - psize) / 2)
                    py += int((self.cell_size - 2 * piece_margin - psize) / 2)

                if shape_bit == 0:  # square
                    if fill_bit == 0:  # solid
                        pygame.draw.rect(self.screen, color, pygame.Rect(px, py, psize, psize))
                    else:  # hollow
                        pygame.draw.rect(self.screen, color, pygame.Rect(px, py, psize, psize), width=4)
                else:  # round
                    center = (px + psize // 2, py + psize // 2)
                    radius = psize // 2
                    if fill_bit == 0:  # solid
                        pygame.draw.circle(self.screen, color, center, radius)
                    else:  # hollow
                        pygame.draw.circle(self.screen, color, center, radius, width=4)