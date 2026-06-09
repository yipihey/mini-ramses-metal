#!/usr/bin/env python3
import argparse, base64, ctypes as C, io, json, os, subprocess
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GAMMA = 1.4

def api(path):
    x=C.CDLL(str(Path(path).resolve()))
    x.ramses_init.restype=C.c_int; x.ramses_init.argtypes=[C.c_char_p,C.c_int]
    x.ramses_get_hydro.restype=C.c_int
    x.ramses_get_hydro.argtypes=[C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_double)]
    x.ramses_amr_step.argtypes=[C.c_int,C.c_int,C.c_int]
    x.ramses_newdt_fine.argtypes=[C.c_int,C.c_int]
    x.ramses_set_dt.argtypes=[C.c_int,C.c_int,C.c_double,C.c_double]
    x.ramses_get_dt.argtypes=[C.c_int,C.c_int,C.POINTER(C.c_double),C.POINTER(C.c_double),C.POINTER(C.c_double)]
    return x

def level_data(lib,h,ndim,lev,nmax=200000):
    nc=1<<ndim; ck=(C.c_int*(ndim*nmax))(); raw={}
    for iv in range(1,6):
        v=(C.c_double*(nc*nmax))(); n=lib.ramses_get_hydro(h,0,iv,lev,nmax,ck,v)
        raw[iv]=np.ctypeslib.as_array(v)[:nc*n].copy()
    if not raw[1].size:return None
    n=len(raw[1])//nc; keys=np.ctypeslib.as_array(ck)[:ndim*n].copy().reshape(n,ndim)
    xy=[]; vals={iv:[] for iv in range(1,6)}
    for j in range(n):
        for c in range(nc):
            xy.append([(2*int(keys[j,d])+((c>>d)&1)+.5)/(2**lev) for d in range(ndim)])
            for iv in range(1,6):vals[iv].append(raw[iv][j*nc+c])
    out={"coords":np.asarray(xy)}
    out.update({f"u{iv}":np.asarray(vals[iv]) for iv in range(1,6)})
    return out

def snap(lib,h,ndim,levels):
    return {l:d for l in levels if (d:=level_data(lib,h,ndim,l)) is not None}

def evolve(lib,nml,ndim,levels,nstep,fixed_dt):
    h=lib.ramses_init(str(nml).encode(),-1)
    if h<=0:raise RuntimeError(f"init failed: {nml}")
    initial=snap(lib,h,ndim,levels); elapsed=0.
    for s in range(1,nstep+1):
        if fixed_dt>0:
            base=levels[0]
            for lev in levels:
                level_dt=fixed_dt/(2**(lev-base))
                lib.ramses_set_dt(h,lev,level_dt,level_dt)
            elapsed+=fixed_dt
        else:
            lib.ramses_newdt_fine(h,levels[0]); a,b,c=C.c_double(),C.c_double(),C.c_double()
            lib.ramses_get_dt(h,levels[0],C.byref(a),C.byref(b),C.byref(c)); elapsed+=max(a.value,0)
        lib.ramses_amr_step(h,levels[0],s)
    return initial,snap(lib,h,ndim,levels),elapsed

def store(path,usual,local,tu,tl,ndim,levels,nstep):
    z={"ndim":ndim,"levels":levels,"nstep":nstep,"t_usual":tu,"t_local":tl}
    for name,s in (("usual",usual),("local",local)):
        for lev,d in s.items():
            for k,v in d.items():z[f"{name}_L{lev}_{k}"]=v
    np.savez_compressed(path,**z)

def run(a):
    os.environ.update(RAMSES_GPU_HYDRO="1",RAMSES_METAL_CACHE="1",RAMSES_METALLIB=str(Path(a.metallib).resolve()))
    lib=api(a.library); prefix="advect" if a.ndim==1 else "sedov"; levels=[5,6] if a.ndim==1 else [4,5,6]
    i1,u,tu=evolve(lib,HERE/f"{prefix}_usual.nml",a.ndim,levels,a.steps,a.fixed_dt)
    i2,l,tl=evolve(lib,HERE/f"{prefix}_localppm.nml",a.ndim,levels,a.steps,a.fixed_dt)
    store(a.output,u,l,tu,tl,a.ndim,levels,a.steps)
    store(Path(a.output).with_name(Path(a.output).stem+"_initial.npz"),i1,i2,0,0,a.ndim,levels,0)
    print("wrote",a.output)

def get(z,name,lev):
    p=f"{name}_L{lev}_";return{k[len(p):]:z[k] for k in z.files if k.startswith(p)}
def prim(d):
    r=d["u1"];u=d["u2"]/r;v=d["u3"]/r;w=d["u4"]/r
    return r,u,v,w,(GAMMA-1)*(d["u5"]-.5*r*(u*u+v*v+w*w))
def style(ax):
    ax.set_facecolor("#10151d");ax.grid(color="#607080",alpha=.17)
    ax.tick_params(colors="#c8d2dc")
    for s in ax.spines.values():s.set_color("#46515e")
    ax.xaxis.label.set_color("#dce5ed");ax.yaxis.label.set_color("#dce5ed");ax.title.set_color("#f4f7fa")
def image(fig):
    b=io.BytesIO();fig.savefig(b,format="png",dpi=170,bbox_inches="tight",facecolor="#10151d");plt.close(fig)
    return "data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()

def lineplot(z,z0):
    def composite(data,name):
        n=2**max(map(int,data["levels"]));a=np.full(n,np.nan)
        for lev in sorted(map(int,data["levels"])):
            d=get(data,name,lev)
            if not d:continue
            rho=prim(d)[0];w=n//(2**lev)
            for x,q in zip(d["coords"][:,0],rho):
                j=int(x*n);a[max(0,j-w//2):min(n,max(0,j-w//2)+w)]=q
        return (np.arange(n)+.5)/n,a
    xu,ru=composite(z,"usual");xl,rl=composite(z,"local");xi,ri=composite(z0,"usual")
    f,ax=plt.subplots(2,1,figsize=(10.5,6.5),sharex=True);f.patch.set_facecolor("#10151d")
    ax[0].plot(xi,ri,"--",color="#7e8994",label="initial");ax[0].plot(xu,ru,color="#65a9ff",lw=2,label="usual: MC + HLLC")
    ax[0].plot(xl,rl,color="#ffb45e",lw=2,label="Local PPM + two-shock");ax[0].legend(frameon=False,ncol=3,labelcolor="#dce5ed")
    ax[0].set(ylabel="density",title="Two-level AMR advection: composite density")
    common=xu;du=ru;dl=rl
    ax[1].plot(common,dl-du,color="#f27474");ax[1].axhline(0,color="#8995a2",lw=.8);ax[1].set(xlabel="x",ylabel="Local - usual")
    for a in ax:style(a);a.axvspan(.25,.51,color="#9b7cff",alpha=.07)
    f.tight_layout()
    return image(f),{"usual_peak":float(ru.max()),"local_peak":float(rl.max()),"usual_tv":float(np.abs(np.diff(ru)).sum()),"local_tv":float(np.abs(np.diff(rl)).sum()),"l1":float(np.abs(dl-du).mean())}

def raster(z,name,field,n=256):
    a=np.full((n,n),np.nan)
    for lev in sorted(map(int,z["levels"])):
        d=get(z,name,lev)
        if not d:continue
        val=prim(d)[0 if field=="rho" else 4];w=max(1,n//(2**lev))
        for (x,y),q in zip(d["coords"],val):
            ix,iy=int(x*n),int(y*n);x0=max(0,ix-w//2);y0=max(0,iy-w//2)
            a[y0:min(n,y0+w),x0:min(n,x0+w)]=q
    return a

def sedovplot(z):
    u,l=raster(z,"usual","rho"),raster(z,"local","rho");d=l-u
    lo,hi=np.nanpercentile(np.r_[u.ravel(),l.ravel()],[1,99.5]);lim=np.nanpercentile(np.abs(d),99)
    f,ax=plt.subplots(1,3,figsize=(13.2,4.4));f.patch.set_facecolor("#10151d")
    for j,(a,q,t) in enumerate(zip(ax,[u,l,d],["Usual: MC + HLLC","Local PPM + two-shock","Local - usual"])):
        im=a.imshow(q,origin="lower",extent=(0,1,0,1),cmap="magma" if j<2 else "coolwarm",vmin=lo if j<2 else -lim,vmax=hi if j<2 else lim)
        a.set(title=t,xlabel="x");style(a);f.colorbar(im,ax=a,fraction=.046,pad=.03)
    ax[0].set_ylabel("y");f.suptitle("2D Sedov blast on three AMR levels",color="#f4f7fa",fontsize=15);f.tight_layout()
    yy,xx=np.indices(u.shape);rr=np.hypot((xx+.5)/256-.5,(yy+.5)/256-.5)
    def scatter(a):
        ids=np.digitize(rr.ravel(),np.linspace(0,.5,70));means=[];std=[]
        for k in range(1,70):
            v=a.ravel()[ids==k];v=v[np.isfinite(v)]
            if len(v):means.append(v.mean());std.append(v.std())
        return float(np.mean(std)/max(np.mean(means),1e-30))
    return image(f),{"usual_peak":float(np.nanmax(u)),"local_peak":float(np.nanmax(l)),"mad":float(np.nanmean(np.abs(d))),"usual_scatter":scatter(u),"local_scatter":scatter(l)}

def radial(z):
    f,a=plt.subplots(figsize=(9.7,5.1));f.patch.set_facecolor("#10151d")
    for name,col,label in [("usual","#65a9ff","usual: MC + HLLC"),("local","#ffb45e","Local PPM + two-shock")]:
        rho=raster(z,name,"rho");yy,xx=np.indices(rho.shape);r=np.hypot((xx+.5)/rho.shape[1]-.5,(yy+.5)/rho.shape[0]-.5)
        edges=np.linspace(0,.5,100);cent=.5*(edges[:-1]+edges[1:]);mean=np.full(len(cent),np.nan);std=np.full(len(cent),np.nan)
        ids=np.digitize(r.ravel(),edges)
        for k in range(1,len(edges)):
            v=rho.ravel()[ids==k];v=v[np.isfinite(v)]
            if len(v):mean[k-1]=v.mean();std[k-1]=v.std()
        a.plot(cent,mean,lw=2,color=col,label=label);a.fill_between(cent,mean-std,mean+std,color=col,alpha=.12)
    a.set(xlabel="radius",ylabel="density",title="Composite radial density mean and 1-sigma scatter");a.legend(frameon=False,labelcolor="#dce5ed");style(a);f.tight_layout()
    return image(f)

def html(a):
    z1=np.load(a.one);z0=np.load(Path(a.one).with_name(Path(a.one).stem+"_initial.npz"));z2=np.load(a.two)
    za=np.load(a.acoustic)
    zamr=np.load(a.acoustic_amr)
    i1,m1=lineplot(z1,z0);i2,m2=sedovplot(z2);i3=radial(z2)
    ia="data:image/png;base64,"+base64.b64encode(Path(a.acoustic).with_suffix(".png").read_bytes()).decode()
    iamr="data:image/png;base64,"+base64.b64encode(Path(a.acoustic_amr).with_suffix(".png").read_bytes()).decode()
    ma=json.loads(str(za["metadata"]))
    mamr=json.loads(str(zamr["metadata"]))
    adv_steps=int(z1["nstep"]);adv_time=float(z1["t_usual"])
    sedov_steps=int(z2["nstep"]);sedov_time=float(z2["t_usual"])
    git=subprocess.check_output(["git","rev-parse","--short","HEAD"],cwd=ROOT,text=True).strip()
    payload=json.dumps({"advection":m1,"sedov":m2})
    text=f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>mini-RAMSES AMR hydro comparison</title>
<style>:root{{--bg:#0b1016;--p:#141b24;--ink:#e7edf3;--muted:#9ba9b7;--blue:#65a9ff;--orange:#ffb45e;--line:#263240}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:46px 24px 70px}}h1{{font-size:clamp(2rem,5vw,4rem);line-height:1.05}}h2{{margin-top:38px}}.lede{{max-width:850px;color:#bdc8d2;font-size:1.15rem}}.blue{{color:var(--blue)}}.orange{{color:var(--orange)}}.muted{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}}.card,.fig,details,.finding{{background:var(--p);border:1px solid var(--line);border-radius:14px;padding:18px}}.finding{{border-color:#8d5933;background:#201912;margin:24px 0}}.metric{{font-size:1.8rem;font-weight:700}}.fig{{margin:16px 0 28px}}img{{width:100%;display:block;border-radius:8px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:.65rem;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{color:#ffca84}}</style></head>
<body><main><div class="muted">mini-RAMSES / Metal hydro / git {git}</div><h1>AMR comparison:<br><span class="blue">usual solver</span> vs <span class="orange">Local PPM</span></h1>
<p class="lede">Matched production AMR runs compare RAMSES's usual monotonized-central reconstruction with HLLC against the new compact one-ghost Local PPM characteristic trace with the two-shock Riemann solver.</p>
<div class="finding"><strong>Current conclusion:</strong> the smooth-wave implementation works on both uniform and fixed AMR meshes. After ten box crossings at exactly t=5, Local PPM retains {ma["localppm"]["amplitude_retention"]:.3f} of the amplitude on the uniform L7 mesh and {mamr["localppm"]["amplitude_retention"]:.3f} through a central L7 patch embedded in L6, with AMR phase error {mamr["localppm"]["phase_error_wavelengths"]:+.4f} wavelengths. The wave remains continuous at both coarse-fine interfaces without obvious reflection or ringing. Correcting AMR subcycling does <em>not</em> fix the 2D Sedov blast: Local PPM still produces a much less developed shock, so the remaining issue is nonlinear strong-shock handling and/or multidimensional AMR flux coupling rather than smooth-wave propagation.</div>
<h2>Uniform-grid acoustic reference test</h2><p class="muted">The exact EnzoNG setup: 128 cells, four wavelengths, amplitude 10<sup>-3</sup>, background u<sub>0</sub>=c<sub>s</sub>=1, and 4,269 equal steps to exactly t=5. The right-going wave travels at 2c<sub>s</sub> and completes ten box crossings.</p><div class="fig"><img src="{ia}"></div>
<table><tr><th>Metric</th><th>Usual</th><th>Local PPM</th></tr><tr><td>Amplitude retention</td><td>{ma["usual"]["amplitude_retention"]:.6f}</td><td>{ma["localppm"]["amplitude_retention"]:.6f}</td></tr><tr><td>Phase error, wavelengths</td><td>{ma["usual"]["phase_error_wavelengths"]:+.6f}</td><td>{ma["localppm"]["phase_error_wavelengths"]:+.6f}</td></tr><tr><td>Harmonic distortion</td><td>{ma["usual"]["harmonic_distortion"]:.6f}</td><td>{ma["localppm"]["harmonic_distortion"]:.6f}</td></tr><tr><td>L1 density error</td><td>{ma["usual"]["l1_density_error"]:.3e}</td><td>{ma["localppm"]["l1_density_error"]:.3e}</td></tr></table>
<h2>Acoustic wave through a fixed fine patch</h2><p class="muted">Static L6 base mesh with one continuous L7 patch from x=0.296875 to 0.71875. The coarse level takes 2,135 steps with dt={mamr["configuration"]["coarse_dt"]:.8f}; each coarse step contains two fine steps with dt={mamr["configuration"]["fine_dt"]:.8f}. Both levels therefore finish at exactly t=5.</p><div class="fig"><img src="{iamr}"></div>
<table><tr><th>Metric</th><th>Usual</th><th>Local PPM</th></tr><tr><td>Amplitude retention</td><td>{mamr["usual"]["amplitude_retention"]:.6f}</td><td>{mamr["localppm"]["amplitude_retention"]:.6f}</td></tr><tr><td>Phase error, wavelengths</td><td>{mamr["usual"]["phase_error_wavelengths"]:+.6f}</td><td>{mamr["localppm"]["phase_error_wavelengths"]:+.6f}</td></tr><tr><td>Harmonic distortion</td><td>{mamr["usual"]["harmonic_distortion"]:.6f}</td><td>{mamr["localppm"]["harmonic_distortion"]:.6f}</td></tr><tr><td>Mass change</td><td>{mamr["usual"]["mass_change"]:.3e}</td><td>{mamr["localppm"]["mass_change"]:.3e}</td></tr></table>
<h2>Nonlinear AMR stress tests</h2><p class="muted">These reruns use corrected AMR subcycling: each finer level takes half the parent timestep while the base level controls the reported physical time.</p>
<div class="grid"><div class="card">Advection peak, usual<div class="metric blue">{m1["usual_peak"]:.4f}</div></div><div class="card">Advection peak, Local PPM<div class="metric orange">{m1["local_peak"]:.4f}</div></div><div class="card">Sedov peak, usual<div class="metric blue">{m2["usual_peak"]:.4f}</div></div><div class="card">Sedov peak, Local PPM<div class="metric orange">{m2["local_peak"]:.4f}</div></div></div>
<h2>1D feature crossing an AMR boundary</h2><p class="muted">Two-level periodic AMR, L5-L6. Both solvers run {adv_steps} base-level steps with dt<sub>L5</sub>=0.004 and dt<sub>L6</sub>=0.002 to t={adv_time:.2f}. The shaded band marks the initially refined region.</p><div class="fig"><img src="{i1}"></div>
<table><tr><th>Metric</th><th>Usual</th><th>Local PPM</th></tr><tr><td>Peak density</td><td>{m1["usual_peak"]:.6f}</td><td>{m1["local_peak"]:.6f}</td></tr><tr><td>Total variation</td><td>{m1["usual_tv"]:.6f}</td><td>{m1["local_tv"]:.6f}</td></tr><tr><td>Mean absolute difference</td><td colspan="2">{m1["l1"]:.3e}</td></tr></table>
<h2>2D Sedov blast on three AMR levels</h2><p class="muted">Static L4-L6 hierarchy around the pressure impulse. Both solvers run {sedov_steps} base-level steps with dt<sub>L4</sub>=2.5e-4, dt<sub>L5</sub>=1.25e-4, and dt<sub>L6</sub>=6.25e-5 to t={sedov_time:.2f}. Coarse cells are drawn first and fine cells overlay them.</p><div class="fig"><img src="{i2}"></div><div class="fig"><img src="{i3}"></div>
<table><tr><th>Metric</th><th>Usual</th><th>Local PPM</th></tr><tr><td>Peak density</td><td>{m2["usual_peak"]:.6f}</td><td>{m2["local_peak"]:.6f}</td></tr><tr><td>Normalized radial scatter</td><td>{m2["usual_scatter"]:.3e}</td><td>{m2["local_scatter"]:.3e}</td></tr><tr><td>Mean absolute difference</td><td colspan="2">{m2["mad"]:.3e}</td></tr></table>
<h2>Reproducibility</h2><details open><summary>Configuration</summary><p>Usual: <code>slope_type=2</code>, <code>riemann='hllc'</code>. Local PPM: <code>slope_type=10</code>, <code>riemann='twoshock'</code>. Both use <code>RAMSES_GPU_HYDRO=1</code> and <code>RAMSES_METAL_CACHE=1</code>, including coarse-fine ghost fill and reflux.</p><p>This is a robustness and visual-quality comparison, not a performance benchmark. Raw metrics: <code>{payload}</code></p></details>
</main></body></html>"""
    Path(a.output).write_text(text);print("wrote",a.output)

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest="cmd",required=True)
    r=s.add_parser("run");r.add_argument("--ndim",type=int,required=True);r.add_argument("--library",required=True);r.add_argument("--metallib",required=True);r.add_argument("--steps",type=int,required=True);r.add_argument("--fixed-dt",type=float,default=0.0);r.add_argument("--output",required=True)
    h=s.add_parser("html");h.add_argument("--one",required=True);h.add_argument("--two",required=True);h.add_argument("--acoustic",required=True);h.add_argument("--acoustic-amr",required=True);h.add_argument("--output",required=True)
    a=p.parse_args();run(a) if a.cmd=="run" else html(a)
if __name__=="__main__":main()
