[1]: https://bitbucket.org/rteyssie/ramses/
[2]: https://bitbucket.org/rteyssie/mini-ramses/

## This fork: a fast GPU GLM-MHD solver ##

The `cuda-dedner-mhd` branch adds a cell-centred, Dedner-cleaned **GLM-MHD** solver to the
CUDA port and optimises it heavily. On magnetised driven turbulence it is **statistically
equivalent to upstream constrained-transport MHD** (same power spectra, Mach number and field
energy to ≈1%) while running **3–5× faster**.

📄 **[GLM-MHD vs CT-MHD turbulence comparison report](docs/glm_vs_ct_turbulence/index.html)**
— throughput, statistical equivalence, power spectra, midplane field slices, and the
investigation that found & fixed an over-aggressive Dedner cleaning speed (the source of
excess small-scale dissipation; now `glm_ch_scale` in `hydro_params`). See
[`runs/mhd_cuda/README.md`](runs/mhd_cuda/README.md) for the full GPU optimisation journey and
per-kernel benchmarks.

---

## mini-ramses ##

The mini-ramses repository is a fork of the [main RAMSES repository][1]. It was created as a stripped-down version of the main code base created in order to facilitate the development of major updates of RAMSES' core routines. This stripped-down version is still available in the `master` branch.

A rapidly-evolving development of mini-ramses based on a completely changed data-structure is contained in the `develop` branch. This branch is work in progress, use it only if you know what you're doing ;)

You can download the code by cloning the git repository using 
```
$ git clone https://bitbucket.org/rteyssie/mini-ramses.git
```

If you want to contribute to mini-ramses, you can either ask me (romain.teyssier@gmail.com) for your personal new branch in this repository which I will give you write access to, or you can fork this repository. To bring changes back into the `develop` branch of mini-ramses, simply issue a pull request.

To compile and execute the standard test cases, please follow these steps.

1- Shock tube test in 1D:

```
$ cd bin
$ make clean
$ make NDIM=1 HYDRO=1
$ cd ..
$ bin/ramses1d namelist/tube1d.nml
```

2- Sedov explosion in 2D:

```
$ cd bin
$ make clean
$ make NDIM=2 HYDRO=1
$ cd ..
$ bin/ramses2d namelist/sedov2d.nml
```

3- Magnetic loop advection in 2D:

```
$ cd bin
$ make clean
$ make NDIM=2 HYDRO=1 MHD=1 INIT=LOOP
$ cd ..
$ bin/ramses2d namelist/loop.nml
```

4- Sedov explosion test in 3D:

```
$ cd bin
$ make clean
$ make NDIM=3 HYDRO=1
$ cd ..
$ bin/ramses3d namelist/sedov3d.nml
```

5- Cosmological N body simulation in 3D

```
$ cd bin
$ make clean
$ make NDIM=3 HYDRO=0 GRAV=1 UNITS=COSMO
$ cd ..
$ utils/scripts/load_cosmo_ic.sh
$ bin/ramses3d namelist/dmo.nml
```

6- Molecular core test in 3D:

```
$ cd bin
$ make clean
$ make NDIM=3 HYDRO=1 GRAV=1 UNITS=COEUR INIT=COEUR
$ cd ..
$ bin/ramses3d namelist/coeur.nml
```

You get the picture now ;-)

### GPU gravity (Apple Silicon / Metal)

The DM particle-mesh gravity stack has a GPU port for Apple Silicon, built with
`make COMPILER=METAL GRAV=1 NDIM=3`. It is validated against the CPU reference to
the floating-point / AMR-chaos floor on the 3D cosmological DMO run. See
[`gpu/metal/STATUS.md`](gpu/metal/STATUS.md) for build/run instructions, the test
suite, and validation evidence (`gpu/metal/PORT_MAP.md` has the per-kernel
parity history).

To visualize the 2D and 3D results, compile the map making executable in the utils/f90 directory.

```
$ cd utils/f90
$ gfortran amr2map.f90 -o amr2map
$ cd ../..
$ utils/f90/amr2map -inp output_00002 -out dens.map -typ 1
$ utils/py/map2img.py dens.map --log
```

In the molecular cloud collapse case, you can also explore the movie1 directory and use the python function directly on any of the maps in there. 
If you have the MPI library properly installed on your system, you can repeat all the tests above using the MPI=1 compilation option.

