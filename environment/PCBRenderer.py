import math

import cairocffi as cairo
import environment.GridCells as gc


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

colours = {0: (0.2588, 0.5294, 0.9608),  # nice blue
           1: (1.0, 0.5, 0.0),  # orange
           2: (1.0, 1.0, 0.0),  # yellow
           3: (1.0, 0.2, 0.8),  # pink
           4: (1.0, 0.0, 0.0),  # red
           5: (0.0, 1.0, 0.0),  # green
           6: (0.0, 0.0, 1.0),  # very blue
           7: (0.8, 0.0, 1.0),  # purple
           8: (0.0, 1.0, 1.0)  # turquoise
           }


def global_to_render(row, col, pixel_height, pixel_width):
    return ((row * 2) + 1) * pixel_height, ((col * 2) + 1) * pixel_width


def get_colour(colour_id):
    return colours.get(colour_id, (1.0, 1.0, 1.0))


def render_net_endpoints(cr, row, col, radius, agent_id):
    (r, g, b) = get_colour(agent_id)
    cr.set_source_rgb(r, g, b)

    cr.arc(col, row, radius, 0, 2 * math.pi)
    cr.stroke()

    # if id is not None and id != -1:
    #     (_, _, text_width, text_height, _, _) = cr.text_extents(str(id))
    #     cr.move_to(col - (text_width / 2), row + (text_height / 2))
    #     cr.show_text(str(id))
    #     cr.new_path()


def render_board(rows, cols, nets, grid):
    ims = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    cr = cairo.Context(ims)

    cr.set_source_rgb(0.0, 0.0, 0.0)
    cr.rectangle(0, 0, width, height)
    cr.fill()

    cr.set_line_join(cairo.LINE_JOIN_ROUND)
    cr.set_line_cap(cairo.LINE_CAP_ROUND)

    cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL,
                        cairo.FONT_WEIGHT_NORMAL)
    cr.set_font_size(40)

    pixel_height = height // (2 * rows)
    pixel_width = width // (2 * cols)

    line_width = min(pixel_height / 8, pixel_width / 8)
    cr.set_line_width(line_width)

    radius = min(pixel_height / 4, pixel_width / 4)

    for net_id, net in nets.items():
        row, col = global_to_render(net.start.row, net.start.col, pixel_height, pixel_width)
        render_net_endpoints(cr, row, col, radius, net.agent_id)

        row, col = global_to_render(net.end.row, net.end.col, pixel_height, pixel_width)
        render_net_endpoints(cr, row, col, radius, net.agent_id)

    for _, net in nets.items():
        if grid[net.start.row, net.start.col] > gc.GridCells.VIA.value:
            cr.new_path()
            (r, g, b) = get_colour(net.agent_id)
            cr.set_source_rgb(r, g, b)
            direction = grid[net.start.row, net.start.col] % gc.GridCells.VIA.value
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
