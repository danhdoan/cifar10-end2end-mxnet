"""
Visualization module
====================
"""


from mxnet import nd

from IPython import display
import matplotlib.pyplot as plt


def show_images(images, nrows, ncols, titles=None, scale=1.5):
    """Show images in grid view
    
    Parameters
    ----------
    images : List[array]
        List of image arrays
    nrows : int 
        Number of rows to display
    ncols : int
        Number of columns to display
    titles : Optional[List[str]]
        List of image titles
    scale : Optional[double]
        Figure scale to display
    """
    figsize = (scale * ncols, scale * nrows)
    _, axes = plt.subplots(nrows, ncols, figsize=figsize)

    axes = axes.flatten()

    for i, (ax, img) in enumerate(zip(axes, images)):
        if isinstance(img, nd.NDArray):
            img = img.asnumpy()

        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        if titles:
            ax.set_title(titles[i])


def plot(X, Y, title=None, xlabel=None, ylabel=None, fmts=None, legend=[],
        xlim=None, ylim=None, xscale='linear', yscale='linear',
        figsize=(6, 4)):
    """Plot 2D graph"""
    plt.rcParams['figure.figsize'] = figsize
    axes = plt.gca()
    if isinstance(X, nd.NDArray): X = X.asnumpy()
    if isinstance(Y, nd.NDArray): Y = Y.asnumpy()
    if not hasattr(X[0], '__len__'): X = [X]
    if not hasattr(Y[0], '__len__'): Y = [Y]
    if len(X) != len(Y): X = X * len(Y)
    if not fmts: fmts = ['-'] * len(X)

    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if isinstance(x, nd.NDArray): x = x.asnumpy()
        if isinstance(y, nd.NDArray): y = y.asnumpy()

        axes.plot(x, y, fmt)

    set_axes(axes, title, xlabel, ylabel, legend, xlim, ylim, xscale, yscale)


def set_axes(axes, title, xlabel, ylabel, legend, xlim, ylim, xscale, yscale):
    """Set attributes to figure"""
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)

    if legend: 
        axes.legend(legend)

    axes.set_xscale(xscale)
    axes.set_yscale(yscale)

    if xlim: 
        axes.set_xlim(xlim)
    if ylim: 
        axes.set_ylim(ylim)

    axes.grid()


class Animator():
    """Animator class"""

    def __init__(self, title=None, xlabel=None, ylabel=None, legend=[],
                xlim=None, ylim=None,
                xscale='linear', yscale='linear',
                fmts=None,
                nrows=1, ncols=1, figsize=(6, 4)):
        """Initialize object instance"""
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1: self.axes = [self.axes]
        
        self.config_axes = lambda : set_axes(
            self.axes[0], title, xlabel, ylabel, legend,
            xlim, ylim, xscale, yscale)
        self.X, self.Y, self.fmts = None, None, fmts
        
    def add(self, x, y):
        """Add new items to animator

        Parameters
        ----------
            x, y
                items to add for visualization
        """
        if not hasattr(y, '__len__'): y = [y]
        n = len(y)
        
        if not hasattr(x, '__len__'): x = [x] * n
        
        if not self.X: self.X = [[] for _ in range(n)]
        if not self.Y: self.Y = [[] for _ in range(n)]
        if not self.fmts: self.fmts = ['-'] * n
            
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
                
        self.axes[0].cla()
        
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
            
        self.config_axes()
        
        display.display(self.fig)
        display.clear_output(wait=True)
        
    def savefig(self, save_path='figure.png'):
        """Save plot
        
        Parameters
        ----------
            save_path : str
                path to save plot
        """
        self.fig.savefig(save_path)
