#!/usr/bin/env python3
"""Forced nonlinear BVP with a reference solution, testing the residual estimate beyond corneal data."""
from pathlib import Path
import numpy as np
import csv
import solver_core as core
OUT=Path(__file__).parent/'validation';OUT.mkdir(exist_ok=True)
a=.7;b=1.3;Lg=2/(3*np.sqrt(3));B=b*Lg;Cp=2/np.pi;CH=np.sqrt(1+Cp**2);kappa=1-B*B/(4*a)
def exact(x):
 u=.4*np.cos(np.pi*x/2)+.1*np.cos(3*np.pi*x/2)
 du=-.4*(np.pi/2)*np.sin(np.pi*x/2)-.1*(3*np.pi/2)*np.sin(3*np.pi*x/2)
 dd=-.4*(np.pi/2)**2*np.cos(np.pi*x/2)-.1*(3*np.pi/2)**2*np.cos(3*np.pi*x/2)
 return u,du,dd
def forcing(x):
 u,d,dd=exact(x);return -dd+a*u-b/np.sqrt(1+d*d)
rows=[];xx=np.linspace(0,1,4001);ur,dr,_=exact(xx)
for N in [6,8,10,12,16,20,24,28,32]:
 xc=.5*(1-np.cos(np.pi*np.arange(N)/(N-1)));K,dK,ddK=core.kernel_factory('rbf',gamma=8.)
 Km=K(xc[:,None],xc[None,:]);D=dK(xc[:,None],xc[None,:]);D2=ddK(xc[:,None],xc[None,:]);r0=dK(0.,xc);r1=K(1.,xc)
 def fj(t):
  c=t[:-1];bias=t[-1];u=Km@c+bias;du=D@c;dd=D2@c
  F=-dd+a*u-b/np.sqrt(1+du*du)-forcing(xc)
  gp=-du*(1+du*du)**(-1.5)
  J=np.column_stack([-D2+a*Km-b*gp[:,None]*D,a*np.ones(N)])
  return np.r_[F,1e3*r0@c,1e3*(r1@c+bias)],np.vstack([J,np.r_[1e3*r0,0],np.r_[1e3*r1,1e3]])
 t=np.zeros(N+1)
 for it in range(80):
  F,J=fj(t)
  if np.linalg.norm(F)<1e-11:break
  step=np.linalg.lstsq(J,-F,rcond=None)[0];lam=1.
  while lam>1e-6:
   tn=t+lam*step
   if np.linalg.norm(fj(tn)[0])<(1-1e-4*lam)*np.linalg.norm(F):t=tn;break
   lam*=.5
  else:break
 c=t[:-1];bias=t[-1];u=K(xx[:,None],xc[None,:])@c+bias;du=dK(xx[:,None],xc[None,:])@c;dd=ddK(xx[:,None],xc[None,:])@c
 rr=-dd+a*u-b/np.sqrt(1+du*du)-forcing(xx);rrn=np.sqrt(np.trapezoid(rr*rr,xx))
 bc0=r0@c;bc1=r1@c+bias;l2=np.sqrt(bc1**2-bc0*bc1+bc0**2/3);lh=np.sqrt(l2*l2+bc0*bc0)
 bound=CH*Cp/kappa*(rrn+a*l2+B*abs(bc0))+lh
 H1=np.sqrt(np.trapezoid((u-ur)**2+(du-dr)**2,xx));rows.append(dict(N=N,H1=H1,H1_bound=bound,effectivity=bound/H1,residual_L2=rrn,bc_derivative=bc0,bc_value=bc1,final_res=np.linalg.norm(fj(t)[0])))
with open(OUT/'manufactured.csv','w',newline='') as h:
 w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(rows)
