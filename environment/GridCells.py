from enum import Enum


class GridCells(Enum):
    BLANK = 20
    N = 1
    NE = 2
    E = 3
    SE = 4
    S = 5
    SW = 6
    W = 7
    NW = 8
    AGENT = 9
    VIA = 10
    VIA_N = 11
    VIA_NE = 12
    VIA_E = 13
    VIA_SE = 14
    VIA_S = 15
    VIA_SW = 16
    VIA_W = 17
    VIA_NW = 18
    OBSTACLE = 0
