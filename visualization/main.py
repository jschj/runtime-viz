import numpy as np
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

if __name__ == '__main__':
    print('Hello world')

    def random_dataset(x, y):
        return hv.Dataset((range(x), range(y), np.random.rand(x, y)), ['x', 'y'], 'test')


    dictonary = {hour: hv.Image(random_dataset(10, 10), ['x', 'y']) for hour in range(10)}
    dictonary2 = {hour: hv.Image(random_dataset(10, 10), ['x', 'y']) for hour in range(10)}
    print(dictonary)
    holomap = hv.HoloMap(dictonary, kdims='Hour')
    holomap2 = hv.HoloMap(dictonary, kdims='Hour')
    layout = holomap + holomap2

    # show(hv.render(holomap))
    hv.renderer('bokeh').save(layout, 'out', fmt='widgets')
