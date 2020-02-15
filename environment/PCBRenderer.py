import cairocffi as cairo

def draw():
    ims = cairo.ImageSurface(cairo.FORMAT_ARGB32, 390, 60)
    cr = cairo.Context(ims)

    cr.set_source_rgb(52, 235, 232)
    cr.set_line_join(cairo.LINE_JOIN_ROUND)

    cr.move_to(10, 50)
    cr.show_text("Disziplin ist Macht.")

    ims.write_to_png("image.png")

draw()