import visdom
import numpy as np
import time


class Visualizer():
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.env = env
        self.index = 1

    def plot_curves(self, d, iters, title='loss', xlabel='iters', ylabel='accuracy'):
        name = list(d.keys())
        val = list(d.values())
        if len(val) == 1:
            y = np.array(val)
        else:
            y = np.array(val).reshape(-1, len(val))
        self.vis.line(Y=y,
                      X=np.array([self.index]),
                      win=title,
                      opts=dict(legend=name, title=title,
                                xlabel=xlabel, ylabel=ylabel),
                      update=None if self.index == 0 else 'append')
        self.index = iters

    def save(self):
        self.vis.save([self.env])


if __name__ == '__main__':
    vis = Visualizer(env='test')
    for i in range(10):
        x = i
        y = 2 * i
        z = 4 * i
        vis.plot_curves({'train': x, 'test': y}, iters=i, title='train')
        vis.plot_curves({'train': z, 'test': y, 'val': i},
                        iters=i, title='test')
        time.sleep(1)
