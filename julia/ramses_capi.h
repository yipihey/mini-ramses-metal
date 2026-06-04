/*===========================================================================
 * ramses_capi.h — C ABI for the mini-RAMSES Julia/C-callable library
 * (libramses<NDIM>d.dylib, built from julia/ramses_capi.f90 + the RAMSES
 * MODOBJ).  Mirror of EnzoModules/src/enzomodules_bridge.h.  Scalars are passed
 * by value; arrays as pointers (column-major / 1-based on the Fortran side).
 * Reuses the POD `Oct` + IDX2/IDX3/IDXP layout macros from gpu/ramses_metal.h.
 *
 * Precision contract: call ramses_precision_bytes() on load and abort on
 * mismatch (npre_bytes: 8=fp64, 4=fp32; ndim: 1/2/3).
 *===========================================================================*/
#ifndef RAMSES_CAPI_H
#define RAMSES_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

/* Precision / shape contract.  Outputs by pointer. */
void ramses_precision_bytes(int *npre_bytes, int *ndim_out, int *twotondim_out);

/* PURE KERNEL: coarse-fine boundary interpolation (interpol_phi).
 * phi_cube/phiold_cube: 3^NDIM coarse-parent cube (1-based cube index).
 * tfrac: subcycle time fraction.  phi_int: 2^NDIM boundary-ghost phi (output). */
void ramses_interpol_phi_kernel(const double *phi_cube, const double *phiold_cube,
                                double tfrac, double *phi_int);

/* ---- State lifecycle (integer handle; 0 = failure) ---------------------- */
int  ramses_init(const char *nml_path, int nrestart);    /* nrestart<0 => use nml value */
void ramses_finalize(int handle);
void ramses_info(int handle, int *levelmin, int *nlevelmax, int *npart, int *nstep_coarse);

/* ---- Particle accessor (idp-keyed; xp/vp column-major n×ndim) ----------- */
void ramses_get_particles(int handle, int n, long long *idp,
                          double *xp, double *vp, int *levelp);

/* ---- Per-routine wrappers for the DMO gravity slice --------------------- */
/* rho/flag/refine auto-route to the GPU when the library was init'd with
 * metal_enabled (the Metal build).  Poisson is explicit: multigrid/phi_fine_cg
 * (CPU) vs ramses_metal_poisson (Metal). */
void ramses_rho_fine(int handle, int ilevel, int rtype);
void ramses_save_phi_old(int handle, int ilevel);
void ramses_multigrid(int handle, int ilevel, int icount);
void ramses_phi_fine_cg(int handle, int ilevel, int icount);
void ramses_force_fine(int handle, int ilevel, int icount);
void ramses_kick_drift(int handle, int ilevel, int action);   /* 1=kick_only 2=kick_drift */
void ramses_flag_fine(int handle, int ilevel, int icount);
void ramses_refine_fine(int handle, int ilevel);
void ramses_metal_poisson(int handle, int ilevel, int icount); /* Metal library only */
void ramses_metal_godunov_fine(int handle, int ilevel);        /* Metal library only */

/* ---- Per-routine wrappers for the HYDRO slice (CPU today; Metal port target) */
/* Conservative state m%uold/m%unew has shape (twotondim,nvar,noct).  Move one
 * variable at a time: field 0=uold 1=unew, ivar in 1..nvar.  ckey is ndim*noct,
 * val is twotondim*noct (same layout as ramses_get_field).  Returns noct. */
int  ramses_nvar(void);
int  ramses_get_hydro(int handle, int field, int ivar, int ilevel, int nmax,
                      int *ckey, double *val);
int  ramses_set_hydro(int handle, int field, int ivar, int ilevel, int n,
                      const int *ckey, const double *val);
void ramses_godunov_fine(int handle, int ilevel);        /* unsplit Godunov solver */
void ramses_set_unew(int handle, int ilevel);            /* unew <- uold */
void ramses_set_uold(int handle, int ilevel);            /* uold <- unew */
void ramses_gravity_hydro_fine(int handle, int ilevel);  /* grav source -> unew */
void ramses_source_hydro_fine(int handle, int ilevel);   /* other source -> unew */
void ramses_synchro_hydro_fine(int handle, int ilevel, double dteff); /* grav source -> uold */
void ramses_upload_fine(int handle, int ilevel);         /* restriction to coarser */
void ramses_cooling_fine(int handle, int ilevel);        /* cooling/heating */
void ramses_newdt_fine(int handle, int ilevel);          /* Courant + particle dt */

#ifdef __cplusplus
}
#endif

#endif /* RAMSES_CAPI_H */
