import numpy as np
from scipy.stats import binned_statistic, binned_statistic_2d
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from cartopy.crs import TransverseMercator, PlateCarree 

def nice_lonlat_gridlines(ax, longitudes=None, latitudes=None, size=8, linewidth=0.5, color='lightgrey', labels = ['bottom', 'left'], **kwargs):
    """
    Makes longitude/latitude ticks nice and small. Cartopys polar stereographic 
    ticks are otherwise annoyingly large and weirdly rotated.

    Based on https://stackoverflow.com/a/65382042
    """

    # Make gridlines
    gl = ax.gridlines(draw_labels=True,x_inline=False,y_inline=False, crs=PlateCarree(), linewidth=linewidth, color=color, **kwargs)

    # Control tick gridline location
    if longitudes is not None:
        gl.xlocator = mticker.FixedLocator(longitudes)
    if latitudes is not None:
        gl.ylocator = mticker.FixedLocator(latitudes)

    # Only show ticks on specified axes
    gl.bottom_labels = 'bottom' in labels
    gl.left_labels   = 'left' in labels
    gl.top_labels    = 'top' in labels
    gl.right_labels  = 'right' in labels

    # Adjust rotation and size of tick labels
    gl.xlabel_style['size']=size
    gl.xlabel_style['rotation']=0
    gl.xlabel_style['ha'] = 'center'   
    gl.xlabel_style['va'] = 'center_baseline'
    
    gl.ylabel_style['size']=size
    gl.ylabel_style['rotation']=90
    gl.ylabel_style['ha'] = 'center'

    return gl

def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3, fontsize=None, textoffset=0, zorder=10):
    """
    Copied and slightly modified from: https://stackoverflow.com/a/35705477
    
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.

    textoffset adds an extra vertical distance between text and bar
    """
    #Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(PlateCarree())
    #Make tmc horizontally centred on the middle of the map,
    #vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = TransverseMercator(sbllx, sblly)
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    #Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    #Calculate a scale bar length if none has been given
    #(Theres probably a more pythonic way of rounding the number but this works)
    if not length: 
        length = (x1 - x0) / 5000 #in km
        ndim = int(np.floor(np.log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf
        #Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']: return int(x)        
            else: return scale_number(x - 10 ** ndim)
        length = scale_number(length) 

    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx, sbx + length * 1000]

    # Save xlim, ylim to later:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    #Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color='k', linewidth=linewidth, zorder=zorder)
    #Plot the scalebar label
    text = ax.text(sbx + length * 500, sby+textoffset, str(length) + ' km', transform=tmc,
                   horizontalalignment='center', verticalalignment='bottom', zorder=zorder)

    # Restore original axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Set font size
    if fontsize is not None:
        text.set_size(fontsize)

def fancy_2d_hist(x,y, values, x_bins, y_bins, statistic = 'count', axes = None,
                  figsize = (6,4), width_ratios = [1, 0.18, 0.05], height_ratios = [0.3,1], 
                  histogram_color = 'slategray', xlabel = '', ylabel = '', 
                  wspace = 0.05, hspace = 0.05, verbose = False,
                  grid_kwargs = {'alpha' : 0.3}, **kwargs):
    """
    Makes a 2d histogram with 1d histograms on sides. 
    x,y, values, x_bins, y_bins, statistic are passed directly to scipy.stats.binned_statistic_2d
    (note that values are not used when statistic = 'count', so that can be set to anything then)

    Optional axes = [ax_2d, ax_1d_x, ax_1d_y, cax]

    If axes is not passed, figsize, width_ratios, height_ratios will be used to create axes. (Third column in figure is for the colorbar.)

    wspace, hspace determines the width and height between subplots.

    grid_kwargs are passed to plt.grid

    **kwargs are passed to plt.pcolormesh (for plotting 2d histogram)

    Returns (fig, axes)
    """

    if axes is None:
        fig = plt.figure(figsize = figsize)
        gs  = gridspec.GridSpec(2,3, figure=fig, width_ratios=width_ratios, height_ratios=height_ratios) 

        ax_2d   = fig.add_subplot(gs[1,0])
        ax_1d_x = fig.add_subplot(gs[0,0])
        ax_1d_y = fig.add_subplot(gs[1,1])
        cax     = fig.add_subplot(gs[1,2])
        axes    = [ax_2d, ax_1d_x, ax_1d_y, cax]
    else:
        ax_2d   = axes[0]
        ax_1d_x = axes[1]
        ax_1d_y = axes[2]
        cax     = axes[3]
    
    # 2D histogram:
    stats2d =  binned_statistic_2d(x, y, values, statistic=statistic, bins=[x_bins, y_bins])
    # Replaces 0 in count with nan
    if statistic == 'count':
        stats2d.statistic[stats2d.statistic == 0] = np.nan
    im = ax_2d.pcolormesh(stats2d.x_edge, stats2d.y_edge, stats2d.statistic.T, **kwargs)
    ax_2d.set_xlabel(xlabel)
    ax_2d.set_ylabel(ylabel)

    # 1D histogram x:
    ax_1d_x.hist(x, bins=stats2d.x_edge, color = histogram_color);
    ax_1d_x.set_ylabel('Count')
    ax_1d_x.set_xlim(stats2d.x_edge[[0,-1]])
    ax_1d_x.xaxis.set_ticklabels([])

    # 1D histogram y:
    ax_1d_y.hist(y, bins=stats2d.y_edge, orientation='horizontal', color = histogram_color);
    ax_1d_y.set_xlabel('Count')
    ax_1d_y.set_ylim(stats2d.y_edge[[0,-1]])
    ax_1d_y.yaxis.set_ticklabels([])

    # Add colorbar
    fig = plt.gcf()
    fig.colorbar(im, cax=cax, label = statistic.capitalize())

    # Gridlines
    ax_2d.grid(**grid_kwargs)
    ax_1d_x.grid(**grid_kwargs)
    ax_1d_y.grid(**grid_kwargs)

    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    if verbose:
        N_total = len(x)
        inside_x_lims = (x >= stats2d.x_edge[0]) & (x <= stats2d.x_edge[-1])
        inside_y_lims = (y >= stats2d.y_edge[0]) & (y <= stats2d.y_edge[-1])
        N_shown = int(sum((inside_x_lims & inside_y_lims)))
        print(f'{N_total - N_shown} points not shown ({100*(N_total - N_shown)/N_total:.2e} %)')
        
    return fig, axes

def binned_statistic_line_plot(xvals, yvals, centers, line = 'mean', shade = 'std', min_nbr_of_points = 10, ax = None,  step = False, **plot_kwargs):
    """

    Line plot based on scipy.stats.binned_statistic. Use for example to plot mean with shaded standard deviation.

    Parameters:
    
        xvals : Values to be binned (x in binned_statistic)
        yvals : The data on which the statistic will be computed (values in binned_statistic)
        centers : center of bins
        line :  statistic passed to binned_statistic (eg 'mean', 'median')
        shade: 'std'/int/None
                std : Shaded area is mean plus/minus one standard deviation
                int : Shaded area cover int % of data. (Example 95 -> shaded 
                      area between 2.5th and 97.5th percentile)
                None: No shaded area
       min_nbr_of_points : minimum number of points ber bin 
       step : If true, plot line as a step plot
       plt_kwargs : passed to matplotlib.plot

    Example:

    xvals = np.random.rand(100)
    yvals = np.random.rand(100)*xvals
    centers = np.linspace(0,1,10)
    
    binned_statistic_line_plot(xvals, yvals, centers, line='mean', shade='95', min_nbr_of_points=2)
    """
    if ax is None:
        ax = plt.gca()

    bin_edges = _get_edges(centers)
    
    nbr_of_points = binned_statistic(xvals, yvals, statistic = 'count', bins = bin_edges).statistic
    enough_points = nbr_of_points >= min_nbr_of_points 
    
    line = binned_statistic(xvals, yvals, statistic = line, bins = bin_edges).statistic
    line[~enough_points] = np.nan
    
    if shade == 'std':
        mean  = binned_statistic(xvals, yvals, statistic = 'mean', bins = bin_edges).statistic
        std   = binned_statistic(xvals, yvals, statistic = 'std', bins = bin_edges).statistic
        lower = line - std
        upper = line + std
    
    if type(shade) == int:
        def percentile_lower(vals):
                return np.percentile(vals, (100-shade)/2)
        def percentile_upper(vals):
                return np.percentile(vals, (100+shade)/2)
        lower = binned_statistic(xvals, yvals, statistic = percentile_lower, bins = bin_edges).statistic
        upper = binned_statistic(xvals, yvals, statistic = percentile_upper, bins = bin_edges).statistic
        lower[~enough_points] = np.nan
        upper[~enough_points] = np.nan
    
    # Plot
    p = _step_plot(bin_edges, line, ax=ax, **plot_kwargs)
        
    color = p[0].get_color()
    if shade is not None:
        ax.fill_between(centers, lower, upper, alpha=0.5, facecolor = color)

    return

def _step_plot(edges, values, label = '', ax = None, **kwargs):
    
    if len(edges) != (len(values)+1):
        raise ValueError('edges should be one element longer than values')
        
    if ax == None:
        ax = plt.gca()
    
    # Add label to first step
    p = ax.plot(edges[0:2], [values[0], values[0]], label=label, **kwargs)
    
    # Plot rest without labels
    color = p[0].get_color()
    kwargs['c'] = color
    for i in range(1,len(values)):
        ax.plot(edges[i:i+2], [values[i], values[i]], label='', **kwargs)

    return p

def _get_edges(centers):
    centers = np.array(centers)
    mid = centers[:-1] + (centers[1:]-centers[:-1])/2
    first = centers[0] - (centers[1]-centers[0])/2
    last  = centers[-1] + (centers[-1]-centers[-2])/2
    return np.concatenate([[first], mid, [last]])

