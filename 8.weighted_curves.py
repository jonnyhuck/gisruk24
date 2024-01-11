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

def fab_idw(d, r):
    """ Calculate fab * IDW Weighting """
    return fab(d, r) * idw(d, r) if d > 0 else 0 

def idw2(d, r):
    """ Calculate IDW-squared Weighting """
    return (1 - (d / r))**2 if d > 0 else 0

def fab_idw2(d, r):
    """ Calculate fab * IDW-squared Weighting """
    return fab(d, r) * idw2(d, r) if d > 0 else 0 

def gauss(d, r, sigma=1):
    """ Calculate Gaussian Weighting """
    return norm.pdf(norm.ppf(d * 0.5 / r + 0.5, loc=0, scale=sigma), loc=0, scale=sigma) if d > 0 else 0 

def fab_gauss(d, r, sigma=1):
    """ Calculate fab * Gaussian Weighting """
    return fab(d, r) * gauss(d, r, sigma) if d > 0 else 0 


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
for f1, f2, colour in [
    # (None, fab, '#e41a1c'), 
    (idw, fab_idw, '#377eb8'), 
    (idw2, fab_idw2, '#4daf4a'), 
    (gauss, fab_gauss, '#984ea3')
    ]:

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
    ax.plot(x, uncorrected, alpha=0.67, color=colour, linestyle='dashed')
    ax.plot(x, corrected, alpha=0.67, color=colour)

# add idea' line
ax.plot(x, x, alpha=0.67, color='#e41a1c', linestyle=(5, (10, 3)))

# add labels
ax.legend([
    # 'Unweighted', 'FAB Unweighted', 
    '$IDW$', '$FAB$ $Corrected$ $IDW$', 
    '$IDW^2$', '$FAB$ $Corrected$ $IDW^2$', 
    "$Gaussian$ $(σ=1)$", '$FAB$ $Corrected$ $Gaussian$ $(σ=1)$',
    '$FAB$ $Corrected$ $Unweighted$', 
    ])

# format plots
ax.grid(True)
ax.set_xlabel('Radius')
ax.set_ylabel('Effective Area')

# output plots
savefig("../Figures/Figure8.png", bbox_inches='tight')