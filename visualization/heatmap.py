import buffer

MAX_RES = 100


class Heatmap:
    def __init__(self, b: buffer.AbstractBuffer, timerange, ax):
        self.b = b

        img = self.b.generate_heatmap(timerange=timerange, max_res=MAX_RES)
        ax.set_title(b.name)
        ax.set_axis_off()
        self.im = ax.imshow(img)

    def update(self, timerange):
        self.im.set_data(self.b.generate_heatmap(timerange=timerange, max_res=MAX_RES))
