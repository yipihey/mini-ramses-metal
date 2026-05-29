#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import FortranFile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file", help="enter filename dens.map")
parser.add_argument("--log", help="plot log variable",action="store_true")
parser.add_argument("--out", help="output a png image")
parser.add_argument("--col", help="choose the color map")
parser.add_argument("--min", help="minimum value", type=float)
parser.add_argument("--max", help="maximum value", type=float)
parser.add_argument("--no-display", help="suppress plot display", action="store_true")
args = parser.parse_args()
print("Reading "+args.file)

# path the the file
path_to_output = args.file

# read image data
with FortranFile(path_to_output, 'r') as f:
    t, dx, dy, dz = f.read_reals('f8')
    nx, ny = f.read_ints('i')
    dat = f.read_reals('f4')

print("nx=",nx,"ny=",ny)
# reshape the output
dat = np.array(dat)
dat = dat.reshape(ny, nx)
dat = np.transpose(dat)
# plot the map
my_dpi = 96
fig, ax = plt.subplots(figsize=(512/my_dpi, 512/my_dpi), dpi=my_dpi)

if args.log:
    dat=np.log10(np.where(dat > 0, dat, np.nan))

col="viridis"
if args.col:
    col=args.col

print("min=",np.nanmin(dat),"max=",np.nanmax(dat))

vmax=None
if args.max:
    vmax=args.max

vmin=None
if args.min:
    vmin=args.min

ax.imshow(dat[:, :].T, interpolation='nearest', origin='lower', cmap=col, vmin=vmin, vmax=vmax)
ax.set_xlabel("nx")
ax.set_ylabel("ny")
if args.out:
    plt.savefig(args.out)
if not args.no_display:
    plt.show()
else:
    plt.close()


