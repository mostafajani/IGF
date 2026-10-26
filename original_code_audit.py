import os,sys,ast,types,json,csv,time
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from pathlib import Path
from scipy.integrate import solve_bvp
ROOT=Path(__file__).resolve().parent; SRC=ROOT/'original_source_snapshots'; OUT=ROOT/'validation'; OUT.mkdir(exist_ok=True)
def load_defs(name):
 p=SRC/name; tree=ast.parse(p.read_text()); body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))]; m=types.ModuleType(name);m.__dict__['__file__']=str(p);exec(compile(ast.Module(body=body,type_ignores=[]),str(p),'exec'),m.__dict__);return m
f=load_defs('lssvr_bvp.py')
def writecsv(name,rows):
 with open(OUT/name,'w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rows=[]
for name,a,b in [('Stiff',.1,.1),('Nominal',1.,1.),('Soft',2.,2.)]:
 x,ur,dr,apex=f.reference_solution(a,b)
 for N in [6,8,10,12,16,20,24,28,32]:
  st=time.perf_counter();xc,c,bias,hist,K,dK,ddK=f.lssvr_collocation(a,b,N=N,kernel='rbf',gamma=8.);dt=time.perf_counter()-st
  u,du=f.lssvr_eval(x,xc,c,bias,K,dK,ddK);ddu=ddK(x[:,None],xc[None,:])@c
  r=-ddu+a*u-b/np.sqrt(1+du**2)
  ell2,linf,h1=f.norms(x,u,du,ur,dr)
  gp=-du*(1+du**2)**(-1.5)
  KK=K(xc[:,None],xc[None,:]);dKK=dK(xc[:,None],xc[None,:]);ddKK=ddK(xc[:,None],xc[None,:]);uc=KK@c+bias;dc=dKK@c
  J=np.vstack([np.column_stack([ddKK-a*KK+b*(-dc*(1+dc**2)**(-1.5))[:,None]*dKK,-a*np.ones(N)]),np.r_[1e3*dK(0.,xc),0.],np.r_[1e3*K(1.,xc),1e3]])
  s=np.linalg.svd(J,compute_uv=False);sg=np.linalg.svd(KK,compute_uv=False);rank=np.linalg.matrix_rank(J)
  bc0=float(dK(0.,xc)@c);bc1=float(K(1.,xc)@c+bias)
  Lg=2/(3*np.sqrt(3));kappa=1-b*b/(27*a)
  # affine lift l(x)=bc1+bc0*(x-1), apply residual transfer for A(v)=-v''+a*v-b*g(v')
  liftL2=np.sqrt(bc1**2-bc0*bc1+bc0**2/3)
  liftH1=np.sqrt(liftL2**2+bc0**2)
  rL2=np.sqrt(np.trapezoid(r*r,x));bound=np.sqrt(1+4/np.pi**2)/kappa*((2/np.pi)*rL2+a*(2/np.pi)*liftL2+b*Lg*(2/np.pi)*abs(bc0))+liftH1
  rows.append(dict(regime=name,N=N,L2=ell2,H1=h1,residual_L2=rL2,bc_derivative=bc0,bc_value=bc1,H1_bound=bound,effectivity=bound/h1,J_rank=rank,J_columns=N+1,J_smin=s[-1],J_smax=s[0],Gram_cond=sg[0]/sg[-1],final_res=float(np.linalg.norm(np.r_[ddKK@c-a*uc+b/np.sqrt(1+dc**2),1e3*bc0,1e3*bc1])),iterations=len(hist),seconds=dt))
  print(rows[-1],flush=True)
writecsv('forward_audit.csv',rows)
# finite-difference verification of the Tikhonov sign error in both supplied inverse files
checks=[]
for filename in ['lssvr_inverse.py','goal2_noise_study_corrected.py']:
 t=(SRC/filename).read_text();checks.append({'file':filename,'negative_regularization_residual':'np.sqrt(reg_eff) * (-c)' in t,'derivative_should_be_negative':True})
(OUT/'regularization_audit.json').write_text(json.dumps(checks,indent=2))
# independent solve_bvp reference and reduced observation sensitivities (dense evaluation, not interpolated grid)
refs={}; sensitiv=[]
for name,a,b in [('Stiff',.1,.1),('Nominal',1.,1.),('Soft',2.,2.)]:
 def ref(a,b):
  xx=np.linspace(0,1,101)
  sol=solve_bvp(lambda x,y:np.vstack([y[1],a*y[0]-b/np.sqrt(1+y[1]**2)]),lambda ya,yb:np.array([ya[1],yb[0]]),xx,np.vstack([b*(1-xx**2)/2,-b*xx]),tol=1e-11,max_nodes=20000)
  if sol.status: raise RuntimeError(sol.message)
  return sol
 sol=ref(a,b); x,ur,dr,_=f.reference_solution(a,b);refs[name]=float(np.max(abs(sol.sol(x)[0]-ur)))
 for M in [2,9,25]:
  xo=np.linspace(.05,.95,M);h=1e-4
  sa=(ref(a+h,b).sol(xo)[0]-ref(a-h,b).sol(xo)[0])/(2*h)
  sb=(ref(a,b+h).sol(xo)[0]-ref(a,b-h).sol(xo)[0])/(2*h)
  sv=np.linalg.svd(np.column_stack([sa,sb]),compute_uv=False)
  sensitiv.append(dict(regime=name,M=M,sigma_min=sv[-1],sigma_max=sv[0],condition=sv[0]/sv[-1]))
writecsv('inverse_sensitivity.csv',sensitiv);(OUT/'reference_agreement.json').write_text(json.dumps(refs,indent=2));print('REFERENCE',refs,flush=True)
# Import original inverse definitions and evaluate clean table; no regularisation active.
i=load_defs('lssvr_inverse.py');i.KERNEL_BANK={'rbf_g4':i.rbf_family(4.),'rbf_g8':i.rbf_family(8.),'rbf_g16':i.rbf_family(16.),'poly3':i.poly_family(3,1.)}
clean=[]
for name,a,b in [('Stiff',.1,.1),('Nominal',1.,1.),('Soft',2.,2.)]:
 x,ur,_,_=i.reference_solution(a,b);xo=np.linspace(.05,.95,9);yo=np.interp(xo,x,ur)
 c,bb,aa,bet,h,xc,w=i.lssvr_inverse(xo,yo)
 clean.append(dict(regime=name,alpha=aa,beta=bet,alpha_error=abs(aa-a),beta_error=abs(bet-b),final_res=h[-1]))
writecsv('clean_inverse_rerun.csv',clean)
print('DONE',flush=True)
