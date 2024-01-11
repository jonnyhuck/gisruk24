"""
* Plot cumulative curves for a range of FAB-corrected and non-corrected functions
"""
from math import hypot
from scipy.stats import norm
from skimage.draw import disk
from numpy import zeros, column_stack, array
from matplotlib.pyplot import subplots, savefig


def fab(d, r):
    """ Calculate FAB Correction """
    return 1 / (d / r) if d > 0 else 0

def idw(d, r):
    """ Calculate IDW Weighting """
    return 1 - (d / r) if d > 0 else 0

# setup
radius = 100
dim = radius * 2
centre = (radius, radius)

# calculate outer ring
outer = disk(centre, radius)

# initialise the landscape and outer ring
landscape_bk = zeros((dim, dim))
landscape_bk[outer] = 1

# plot
fig, ax = subplots(1, 1, figsize=(8, 8))

# loop through each pair of functions
for f1, f2 in [(None, fab)]:

    # init
    x = [0]
    uncorrected = [0]
    corrected = [0]

    # increment 0.1-1 in 0.1 increments
    for inner_proportion in array(range(1, 51, 1)) / 50:

        # record x value
        x.append(inner_proportion)

        # refresh landscape for uncorrected and corrected versions
        uc_surf = landscape_bk.copy()
        c_surf = landscape_bk.copy()

        # calculate inner ring
        inner = disk(centre, radius * inner_proportion)

        # now apply weighting to entire disk
        for r, c in column_stack(outer):
            dist = hypot(c - centre[1], r - centre[0])

            # apply uncorrected weighting
            if f1 is not None:
                uc_surf[(r,c)] = f1(dist, radius)
                
            # apply corrected weighting
            c_surf[(r,c)] = f2(dist, radius)

        # mean value
        uncorrected.append(sum(uc_surf[inner]) / sum(uc_surf[outer]))
        corrected.append(sum(c_surf[inner]) / sum(c_surf[outer]))
        
    # add results from each function
    ax.plot(x, uncorrected, alpha=0.67, color='#377eb8')
    ax.plot(x, x, alpha=0.67, color='#4daf4a')

# add labels
ax.legend([ 'Each $location$ $(cell)$ equally important', 'Each $distance$ equally important'])

# format plots
ax.grid(True)
ax.set_xlabel('Radius')
ax.set_ylabel('Effective Area')

# output plots
savefig("../Figures/Figure2.png", bbox_inches='tight')