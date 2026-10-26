#!/usr/bin/env python3
import os,time,csv,json,platform
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import solve_bvp
from numpy.polynomial.chebyshev import chebvander,chebder,chebval
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import solver_core as core
OUT=Path(__file__).parent/'validation';OUT.mkdir(exist_ok=True)
REG=[('Stiff',.1,.1),('Nominal',1.,1.),('Soft',2.,2.)]
def csvout(name,rows):
 with open(OUT/name,'w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def reference(a,b):
 x=np.linspace(0,1,101)
 s=solve_bvp(lambda x,y:np.vstack([y[1],a*y[0]-b/np.sqrt(1+y[1]**2)]),lambda ya,yb:np.array([ya[1],yb[0]]),x,np.vstack([b*(1-x*x)/2,-b*x]),tol=1e-11,max_nodes=30000)
 if s.status:raise RuntimeError(s.message)
 return s
# Exact Chebyshev coefficient BCs, ODE collocation at interior Lobatto nodes.
def chebyshev_solve(a,b,degree):
 j=np.arange(degree+1);xi=.5*(1-np.cos(np.pi*j/degree));t=2*xi-1
 def matrices(xx):
  tt=2*xx-1;V=chebvander(tt,degree);V1=np.zeros_like(V);V2=np.zeros_like(V)
  for k in range(degree+1):
   c=np.eye(degree+1)[k];V1[:,k]=2*chebval(tt,chebder(c));V2[:,k]=4*chebval(tt,chebder(c,2))
  return V,V1,V2
 V,V1,V2=matrices(xi)
 def residual(c):
  u=V@c;d=V1@c;dd=V2@c;F=dd-a*u+b/np.sqrt(1+d*d)
  J=V2-a*V+b*(-d*(1+d*d)**(-1.5))[:,None]*V1
  F[0]=d[0];J[0]=V1[0];F[-1]=u[-1];J[-1]=V[-1]
  return F,J
 c=np.zeros(degree+1)
 for _ in range(25):
  F,J=residual(c)
  if np.linalg.norm(F,np.inf)<5e-12:break
  step=np.linalg.solve(J,-F);lam=1.
  while lam>1e-8:
   trial=c+lam*step
   if np.linalg.norm(residual(trial)[0])<np.linalg.norm(F):c=trial;break
   lam*=.5
  else:break
 return c,matrices
rows=[];x=np.linspace(0,1,2001)
for name,a,b in REG:
 ref=reference(a,b);ur,dr=ref.sol(x)
 for degree in [8,12,16,24,32]:
  times=[]
  for _ in range(5):
   st=time.perf_counter();c,mat=chebyshev_solve(a,b,degree);times.append(time.perf_counter()-st)
  V,V1,V2=mat(x);u=V@c;du=V1@c
  rows.append(dict(regime=name,degree=degree,nodal_dof=degree+1,L2=np.sqrt(np.trapezoid((u-ur)**2,x)),H1=np.sqrt(np.trapezoid((u-ur)**2+(du-dr)**2,x)),median_seconds=np.median(times)))
csvout('chebyshev_baseline.csv',rows)
# Corrected inverse experiments: residual +sqrt(lambda)c and positive matching Jacobian.
# Explicit relative-noise convention; does not claim statistical inverse-variance weights in raw units.
rng=np.random.default_rng(7);noiserows=[];raw=[]
for name,a,b in REG:
 ref=reference(a,b);xo=np.linspace(.05,.95,9);yc=ref.sol(xo)[0];scale=np.max(abs(yc))
 for sig in [0.,.02,.05,.10]:
  estimates=[]
  for rep in range(20 if sig else 1):
   yo=yc+rng.normal(0,sig*scale,9)
   c,bb,aa,be,h,xc,w=core.lssvr_inverse(xo,yo,noise_std=sig,max_gn=80)
   estimates.append([abs(aa-a)/a,abs(be-b)/b]);raw.append(dict(regime=name,sigma=sig,rep=rep,alpha=aa,beta=be,final_res=h[-1],iterations=len(h)))
  ee=np.array(estimates)
  noiserows.append(dict(regime=name,sigma=sig,alpha_mean=np.mean(ee[:,0]),alpha_std=np.std(ee[:,0]),beta_mean=np.mean(ee[:,1]),beta_std=np.std(ee[:,1])))
  print('NOISE',noiserows[-1],flush=True)
csvout('corrected_noise.csv',noiserows);csvout('corrected_noise_runs.csv',raw)
# Independent seed stream for repeatable bootstrap results. B=200; no claim of validated coverage.
rng=np.random.default_rng(107);ref=reference(1.,1.);xo=np.linspace(.05,.95,9);yc=ref.sol(xo)[0];scale=max(abs(yc));boot=[]
for sig in [0.,.02,.05,.10]:
 yo=yc+rng.normal(0,sig*scale,9)
 c,bb,aa,be,h,xc,w=core.lssvr_inverse(xo,yo,noise_std=sig,max_gn=80)
 fit=core.profile(xo,xc,c,bb,w);res=yo-fit;res-=res.mean();est=[]
 for rep in range(200):
  yy=fit+rng.choice(res,9,replace=True)
  _,_,a2,b2,h2,_,_=core.lssvr_inverse(xo,yy,noise_std=sig,max_gn=80,alpha0=aa,beta0=be)
  est.append([a2,b2])
 est=np.array(est);lo,hi=np.percentile(est,[2.5,97.5],axis=0)
 boot.append(dict(sigma=sig,alpha=aa,beta=be,alpha_lo=lo[0],alpha_hi=hi[0],beta_lo=lo[1],beta_hi=hi[1],B=200))
 print('BOOT',boot[-1],flush=True)
csvout('corrected_bootstrap.csv',boot)
fig,ax=plt.subplots(figsize=(6.4,4.4));ns=[r['sigma'] for r in boot]
ax.plot(ns,[r['alpha_hi']-r['alpha_lo'] for r in boot],'o-',label=r'CI width $\alpha$');ax.plot(ns,[r['beta_hi']-r['beta_lo'] for r in boot],'s--',label=r'CI width $\beta$');ax.set(xlabel='Relative noise fraction',ylabel='Percentile interval width',title='Corrected residual bootstrap (B = 200)');ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(OUT/'corrected_bootstrap.png',dpi=200);plt.close(fig)
# Corrected single-RBF profile; old MKL profile is preserved separately in main manuscript.
rng=np.random.default_rng(207);sig=.05;yo=yc+rng.normal(0,sig*scale,9)
c,bb,aa,be,h,xc,w=core.lssvr_inverse(xo,yo,noise_std=sig,max_gn=80)
fig,ax=plt.subplots(figsize=(6.6,4.4));ax.plot(x,ref.sol(x)[0],'k-',label='Independent BVP reference');ax.plot(x,core.profile(x,xc,c,bb,w),'r--',label=rf'Corrected RBF ($\hat\alpha={aa:.3f},\hat\beta={be:.3f}$)');ax.plot(xo,yo,'bo',label='5% noisy observations');ax.set(xlabel='x',ylabel='u(x)',title='Corrected regularized inverse fit');ax.grid(alpha=.3);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'corrected_profile.png',dpi=200);plt.close(fig)
# Numerical check of corrected Tikhonov residual derivative.
c=np.linspace(-.2,.2,24);lam=.025;eps=1e-6
FD=np.column_stack([(np.sqrt(lam)*(c+eps*np.eye(24)[k])-np.sqrt(lam)*(c-eps*np.eye(24)[k]))/(2*eps) for k in range(24)])
(OUT/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),tikhonov_jacobian_max_error=float(np.max(abs(FD-np.sqrt(lam)*np.eye(24))))),indent=2))
print('DONE',flush=True)
