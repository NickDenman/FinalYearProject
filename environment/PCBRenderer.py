import math

import cairocffi as cairo

from environment.PCB_Board_2D_Static import GridCells

width = 1024
height = 1024
actions = {1: [-1, 0],
           2: [-1, 1],
           3: [0, 1],
           4: [1, 1],
           5: [1, 0],
           6: [1, -1],
           7: [0, -1],
           8: [-1, -1],
           10: [0, 0]}


def global_to_render(row, col, pixel_height, pixel_width):
    return ((row * 2) + 1) * pixel_height, ((col * 2) + 1) * pixel_width


def render_board(rows, cols, nets, grid):
    ims = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    cr = cairo.Context(ims)

    cr.set_source_rgb(0.0, 0.0, 0.0)
    cr.rectangle(0, 0, width, height)
    cr.fill()

    cr.set_source_rgb(52.0 / 255.0, 235.0 / 255.0, 232.0 / 255.0)
    cr.set_line_join(cairo.LINE_JOIN_ROUND)
    cr.set_line_cap(cairo.LINE_CAP_ROUND)

    pixel_height = height // (2 * rows)
    pixel_width = width // (2 * cols)

    line_width = min(pixel_height / 8, pixel_width / 8)
    cr.set_line_width(line_width)

    radius = min(pixel_height / 4, pixel_width / 4)

    for net in nets:
        row, col = global_to_render(net.start.row, net.start.col, pixel_height, pixel_width)
        cr.arc(col, row, radius, 0, 2 * math.pi)
        cr.stroke()

        # if net.agent_id != -1:
        #     cr.move_to(col, row)
        #     cr.show_text(str(net.agent_id))

        row, col = global_to_render(net.end.row, net.end.col, pixel_height, pixel_width)
        cr.arc(col, row, radius, 0, 2 * math.pi)
        cr.stroke()

    for net in nets:
        if grid[net.start.row, net.start.col] > GridCells.VIA.value:
            direction = grid[net.start.row, net.start.col] % GridCells.VIA.value
            row = net.start.row
            col = net.start.col
            render_row, render_col = global_to_render(row, col, pixel_height, pixel_width)

            deltas = actions.get(direction)
            h = math.hypot(deltas[0], deltas[1])
            start_row = render_row + radius * deltas[0] / h
            start_col = render_col + radius * deltas[1] / h

            cr.move_to(start_col, start_row)
            while grid[row, col] != 0 and grid[row, col] != 9 and not (row == net.end.row and col == net.end.col):
                row += deltas[0]
                col += deltas[1]
                render_row, render_col = global_to_render(row, col, pixel_height, pixel_width)

                if row == net.end.row and col == net.end.col:
                    h = math.hypot(deltas[0], deltas[1])
                    render_row -= radius * deltas[0] / h
                    render_col -= radius * deltas[1] / h

                cr.line_to(render_col, render_row)
                deltas = actions.get(grid[row, col])

            cr.stroke()
    ims.write_to_png("image.png")
