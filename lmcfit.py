import numpy as np
import pylab as plt
import pandas as pd
import matplotlib
from mpl_toolkits.mplot3d import axes3d
import io
from astropy.table import vstack,Table
import matplotlib.pyplot as plt
from matplotlib import cm
from astropy import units as u
from lmfit import Minimizer, Parameters, report_fit


# This class initializes the parameters to the best-fit model from the Hydra data from Olsen et al. (2011)
class lmcModel():

    """
    Store LMC model parameters
    """
    
    def __init__(self,vt=0.,tht=0.,
            aro=1.4294247*u.rad,
            dro=-1.2194616*u.rad,
            vsys=263.15027*u.km/u.s,
            didt=-183.95287*u.deg/u.Gyr,
            #th=0.0*u.deg,
            th=141.79040*u.deg,
            v0=87.294243*u.km/u.s,
            inc=34.7*u.deg,
            d0=50.12e3*u.pc,
            eta=0.047051836,
            s=1.,
            mun=0.229*u.mas/u.yr,
            muw=-1.910*u.mas/u.yr,
            dthdt=0.0*u.deg/u.Gyr,
        ):
        
        self.aro=aro # RA of center of rotation in radians
        self.dro=dro # DEC of center of rotation in radians
        self.vsys=vsys # Line of sight velocity of center in km/s
        self.didt=didt # Tumble rate of disk in deg/Gyr
        self.th=th # Position angle of line of nodes measured in degrees E of N
        self.v0=v0 # Peak velocity of rotation curve
        self.inc=inc # Inclination of disk in deg, 0 = face-on
        self.d0=d0 # Distance to center in pc
        self.eta=eta # Angular radius at which rotation curve reaches peak velocity (radians)
        self.r0=eta*d0 # Radius at which rotation curve reaches peak velocity (pc)
        self.s=s # Spin direction as observed on sky, either +/-1 (clockwise/counter-clockwise)
        self.mun=mun # Proper motion of center in N direction, mas/yr
        self.muw=muw # Proper motion of center in W direction, mas/yr
        self.dthdt=dthdt # Precession rate of disk, deg/Gyr
        self.vt=(np.sqrt(mun**2+muw**2)*d0).to(u.rad*u.km/u.s).value*u.km/u.s 
                # Total transverse velocity of center in km/s
        self.tht=np.arctan(-muw/mun).to(u.deg) # Angle of direction of transverse motion of center, deg E of N
        
    def __str__(self):
        ret_str = ""
        ret_str = ret_str+"aro="+str(self.aro)+"\n"
        ret_str = ret_str+"dro="+str(self.dro)+"\n"
        ret_str = ret_str+"vsys="+str(self.vsys)+"\n"
        ret_str = ret_str+"didt="+str(self.didt)+"\n"
        ret_str = ret_str+"th="+str(self.th)+"\n"
        ret_str = ret_str+"v0="+str(self.v0)+"\n"
        ret_str = ret_str+"inc="+str(self.inc)+"\n"
        ret_str = ret_str+"d0="+str(self.d0)+"\n"
        ret_str = ret_str+"eta="+str(self.eta)+"\n"
        ret_str = ret_str+"r0="+str(self.r0)+"\n"
        ret_str = ret_str+"s="+str(self.s)+"\n"
        ret_str = ret_str+"mun="+str(self.mun)+"\n"
        ret_str = ret_str+"muw="+str(self.muw)+"\n"
        ret_str = ret_str+"dthdt="+str(self.dthdt)+"\n"
        ret_str = ret_str+"vt="+str(self.vt)+"\n"
        ret_str = ret_str+"tht="+str(self.tht)+"\n"
        
        return ret_str

# Function to compute all observed velocities and proper motions given a set of model parameters
# Equations from van der Marel et al. (2002)
def modvel(lmcTable,rakey='RA',deckey='DEC',params=None):

    # Observed positions
    try:
        ra = lmcTable[rakey].to(u.deg)
    except:
        ra = lmcTable[rakey]*u.deg

    try:
        dec = lmcTable[deckey].to(u.deg)
    except:
        dec = lmcTable[deckey]*u.deg

    # Model parameters
    if params is None:
        lmcmod=lmcModel()
        aro=lmcmod.aro
        dro=lmcmod.dro
        vsys=lmcmod.vsys
        didt=lmcmod.didt
        th=lmcmod.th
        v0=lmcmod.v0
        inc=lmcmod.inc
        d0=lmcmod.d0
        r0=lmcmod.r0
        s=lmcmod.s
        mun=lmcmod.mun
        muw=lmcmod.muw
        dthdt=lmcmod.dthdt
        vt=np.sqrt(mun**2+muw**2)*d0
        vt=lmcmod.vt
        tht=lmcmod.tht

    else:
        aro = params['aro']*u.rad
        dro = params['dro']*u.rad
        vsys = params['vsys']*u.km/u.s
        didt = params['didt']*u.deg/u.Gyr
        th = params['th']*u.deg
        v0 = params['v0']*u.km/u.s
        inc = params['inc']*u.deg
        d0 = params['d0']*u.pc
        eta = params['eta']
        s = params['s']
        mun = params['mun']*u.mas/u.yr
        muw = params['muw']*u.mas/u.yr
        dthdt = params['dthdt']*u.deg/u.Gyr

        # Derived parameters
        tht = np.arctan(-muw/mun).to(u.deg)
        vt = (np.sqrt(mun**2+muw**2)*d0).to(u.rad*u.km/u.s).value*u.km/u.s
        r0 = eta*d0
        
    ar=ra.to(u.rad)
    dr=dec.to(u.rad)
    ar[ar-aro > np.pi*u.rad] = ar[ar-aro > np.pi*u.rad] - 2*np.pi*u.rad # to set ar-aro between -pi,pi

    muc=-muw*np.sin(th) + mun*np.cos(th)
    mus=-muw*np.cos(th) - mun*np.sin(th)
    wts=d0.to(u.km)*(didt.to(u.rad/u.s) + mus.to(u.rad/u.s))
    wts=wts.value*u.km/u.s
    vtc=d0.to(u.km)*muc.to(u.rad/u.s)
    vtc=vtc.value*u.km/u.s

    rho=np.arccos((np.cos(dr)*np.cos(dro)*np.cos(ar-aro)+np.sin(dr)*np.sin(dro)))
    tphi=(np.sin(dr)*np.cos(dro)-np.cos(dr)*np.sin(dro)*np.cos(ar-aro))/(-np.cos(dr)*np.sin(ar-aro))
    tphi[np.isnan(tphi)]=0
    q2=((aro-ar) < 0) & ((dr-dro) > 0)
    q3=((aro-ar) < 0) & ((dr-dro) < 0)
    q4=((aro-ar) > 0) & ((dr-dro) < 0)
    phi=np.arctan(tphi)

    if (len(phi[q2])>0): phi[q2]=phi[q2]+np.pi*u.rad
    if (len(phi[q3])>0): phi[q3]=phi[q3]+np.pi*u.rad
    if (len(phi[q4])>0): phi[q4]=phi[q4]+2*np.pi*u.rad
    bphi=phi-np.pi/2*u.rad
    neg=(bphi < 0)
    if (len(bphi[neg])>0): bphi[neg]=bphi[neg]+2*np.pi*u.rad

    ff=(np.cos(inc)*np.cos(rho)-np.sin(inc)*np.sin(rho)*np.sin(bphi-th))/np.sqrt(np.cos(inc)**2*np.cos(bphi-th)**2+np.sin(bphi-th)**2)
    rpri=d0*np.sin(rho)/ff
    vr=(v0/r0*rpri)
    vr=np.clip(vr,0,v0)

    # model line of sight velocities
    mv1=vsys*np.cos(rho) # contribution from center of mass motion
    mv2=wts*np.sin(rho)*np.sin(bphi-th) # contribution from transverse motion, component 1
    mv3=(vtc*np.sin(rho))*np.cos(bphi-th) # contribution from transverse motion, component 2
    vrfac=(-s*ff*np.sin(inc))*np.cos(bphi-th)
    mv4=(-s*ff*vr*np.sin(inc))*np.cos(bphi-th) # contribution from internal rotation
    modv=mv1+mv2+mv3+mv4 # total line of sight velocity of model

    # model proper motions
    # broken up by contribution from different sources; 
    # labels 1,2,and 3 refer to x,y,z directions

    # Center-of-mass motion contribution
    mv2cm=vt*np.cos(rho)*np.cos(bphi-tht)-vsys*np.sin(rho)
    mv3cm=-vt*np.sin(bphi-tht)

    div1=np.cos(inc)*np.cos(rho)-np.sin(inc)*np.sin(rho)*np.sin(bphi-th)

    # Contribution from precession and nutation
    mv2pn=d0.to(u.km)*np.sin(rho)/div1*didt.to(u.rad/u.s)*np.sin(bphi-th)*(-np.cos(inc)*np.sin(rho)-np.sin(inc)*np.cos(rho)*np.sin(bphi-th))
    mv2pn=mv2pn.value*u.km/u.s
    mv3pn=d0.to(u.km)*np.sin(rho)/div1*(didt.to(u.rad/u.s)*np.sin(bphi-th)*(-np.sin(inc)*np.cos(bphi-th))+dthdt.to(u.rad/u.s)*np.cos(inc))
    mv3pn=mv3pn.value*u.km/u.s

    # Contribution from internal rotation
    ff2=(np.cos(inc)*np.sin(rho)+np.sin(inc)*np.cos(rho)*np.sin(bphi-th))/np.sqrt(np.cos(inc)**2*np.cos(bphi-th)**2+np.sin(bphi-th)**2)
    mv2int=s*ff2*vr*np.sin(inc)*np.cos(bphi-th)

    ff3=(np.cos(inc)**2*np.cos(bphi-th)**2+np.sin(bphi-th)**2)/np.sqrt(np.cos(inc)**2*np.cos(bphi-th)**2+np.sin(bphi-th)**2)
    mv3int=-s*vr*ff3

    vrfacpm = np.sqrt(ff2**2*np.sin(inc)**2*np.cos(bphi-th)**2 + ff3**2)
    
    cosg=(np.sin(dr)*np.cos(dro)*np.cos(ar-aro)-np.cos(dr)*np.sin(dro))/np.sin(rho)
    sing=(np.cos(dro)*np.sin(ar-aro))/np.sin(rho)
    cosg[np.isnan(cosg)]=0.
    sing[np.isnan(sing)]=-1.

    # Total transverse (proper motion) velocities
    mv2pm=mv2cm+mv2pn+mv2int
    mv3pm=mv3cm+mv3pn+mv3int

    # Transform to observed proper motions in W and N directions
    c1=(np.cos(inc)*np.cos(rho)-np.sin(inc)*np.sin(rho)*np.sin(bphi-th))/(d0.to(u.km)*np.cos(inc))
    muwmod=(c1*(-mv2pm*sing-mv3pm*cosg)*u.rad).to(u.mas/u.yr)
    munmod=(c1*(mv2pm*cosg-mv3pm*sing)*u.rad).to(u.mas/u.yr)

    muwcm=(c1*(-mv2cm*sing-mv3cm*cosg)*u.rad).to(u.mas/u.yr)
    muncm=(c1*(mv2cm*cosg-mv3cm*sing)*u.rad).to(u.mas/u.yr)

    muwpn=(c1*(-mv2pn*sing-mv3pn*cosg)*u.rad).to(u.mas/u.yr)
    munpn=(c1*(mv2pn*cosg-mv3pn*sing)*u.rad).to(u.mas/u.yr)

    muwint=(c1*(-mv2int*sing-mv3int*cosg)*u.rad).to(u.mas/u.yr)
    munint=(c1*(mv2int*cosg-mv3int*sing)*u.rad).to(u.mas/u.yr)

    pmint_fac = (c1*vrfacpm).to(1./u.pc)*(d0.to(u.pc))

    # Store model proper motionas and line of sight velocities
    # (including separate contributions) in data table
    lmcTable['MUNMOD']=munmod
    lmcTable['MUWMOD']=muwmod
    lmcTable['MUNCM']=muncm
    lmcTable['MUWCM']=muwcm
    lmcTable['MUNPN']=munpn
    lmcTable['MUWPN']=muwpn
    lmcTable['MUNINT']=munint
    lmcTable['MUWINT']=muwint

    lmcTable['mv2cm']=mv2cm
    lmcTable['mv2pn']=mv2pn
    lmcTable['mv2int']=mv2int
    lmcTable['mv2pm']=mv2pm

    lmcTable['mv3cm']=mv3cm
    lmcTable['mv3pn']=mv3pn
    lmcTable['mv3int']=mv3int
    lmcTable['mv3pm']=mv3pm
    
    lmcTable['mv1']=mv1
    lmcTable['mv2']=mv2
    lmcTable['mv3']=mv3
    lmcTable['mv4']=mv4
    lmcTable['mvtot']=mv1+mv2+mv3+mv4

    # Geometric quantities useful for later
    lmcTable['rho']=rho
    lmcTable['bphi']=bphi
    lmcTable['c1']=c1
    lmcTable['cosg']=cosg
    lmcTable['sing']=sing
    lmcTable['vrfac']=vrfac
    lmcTable['vrfacpm']=vrfacpm
    lmcTable['pmint_fac']=pmint_fac
    lmcTable['RPRI']=rpri.to(u.kpc)
    lmcTable['VRAD']=vr
    
    
# Function to compute 3-D velocities in LMC plane coordinates given positions, velocities, 
# and proper motions
# Function to compute 3-D velocities in LMC plane coordinates given positions, velocities, 
# and proper motions
def deproject(ra,dec,pmw,pmn,vhel,pmwerr=None,pmnerr=None,vherr=None,params=None,polar=False):

    try:
        u1=ra.unit
        ra.to(u.deg)
    except:
        ra=ra*u.deg
    try:
        u1=dec.unit
        dec.to(u.deg)
    except:
        dec=dec*u.deg
    try:
        u1=pmw.unit
        pmw.to(u.mas/u.yr)
    except:
        pmw=pmw*u.mas/u.yr
    try:
        u1=pmn.unit
        pmn.to(u.mas/u.yr)
    except:
        pmn=pmn*u.mas/u.yr
    try:
        u1=vhel.unit
        vhel.to(u.km/u.s)
    except:
        vhel=vhel*u.km/u.s

    if (pmwerr is not None) and (pmnerr is not None) and (vherr is not None):       
        try:
            u1=pmwerr.unit
            pmwerr.to(u.mas/u.yr)
        except:
            pmwerr=pmwerr*u.mas/u.yr
        try:
            u1=pmnerr.unit
            pmnerr.to(u.mas/u.yr)
        except:
            pmnerr=pmnerr*u.mas/u.yr
        try:
            u1=vherr.unit
            vherr.to(u.km/u.s)
        except:
            vherr=vherr*u.km/u.s

    # Model parameters
    if params is None:
        lmcmod=lmcModel()
        aro=lmcmod.aro
        dro=lmcmod.dro
        vsys=lmcmod.vsys
        didt=lmcmod.didt
        th=lmcmod.th
        v0=lmcmod.v0
        inc=lmcmod.inc
        d0=lmcmod.d0
        r0=lmcmod.r0
        s=lmcmod.s
        mun=lmcmod.mun
        muw=lmcmod.muw
        dthdt=lmcmod.dthdt
        vt=np.sqrt(mun**2+muw**2)*d0
        vt=lmcmod.vt
        tht=lmcmod.tht

    else:
        aro = params['aro']*u.rad
        dro = params['dro']*u.rad
        vsys = params['vsys']*u.km/u.s
        didt = params['didt']*u.deg/u.Gyr
        th = params['th']*u.deg
        v0 = params['v0']*u.km/u.s
        inc = params['inc']*u.deg
        d0 = params['d0']*u.pc
        eta = params['eta']
        s = params['s']
        mun = params['mun']*u.mas/u.yr
        muw = params['muw']*u.mas/u.yr
        dthdt = params['dthdt']*u.deg/u.Gyr

        # Derived parameters
        tht = np.arctan(-muw/mun).to(u.deg)
        vt = (np.sqrt(mun**2+muw**2)*d0).to(u.rad*u.km/u.s).value*u.km/u.s
        r0 = eta*d0
    
    ar=ra.to(u.rad)
    dr=dec.to(u.rad)
    ar[ar-aro > np.pi*u.rad] = ar[ar-aro > np.pi*u.rad] - 2*np.pi*u.rad # to set ar-aro between -pi,pi
    ar[ar-aro < -np.pi*u.rad] = ar[ar-aro < -np.pi*u.rad] + 2*np.pi*u.rad # to set ar-aro between -pi,pi
    
    rho=np.arccos((np.cos(dr)*np.cos(dro)*np.cos(ar-aro)+np.sin(dr)*np.sin(dro)))
    tphi=(np.sin(dr)*np.cos(dro)-np.cos(dr)*np.sin(dro)*np.cos(ar-aro))/(-np.cos(dr)*np.sin(ar-aro))
    tphi[np.isnan(tphi)]=0
    q2=((aro-ar) < 0) & ((dr-dro) > 0)
    q3=((aro-ar) < 0) & ((dr-dro) < 0)
    q4=((aro-ar) > 0) & ((dr-dro) < 0)
    phi=np.arctan(tphi)

    if (len(phi[q2])>0): phi[q2]=phi[q2]+np.pi*u.rad
    if (len(phi[q3])>0): phi[q3]=phi[q3]+np.pi*u.rad
    if (len(phi[q4])>0): phi[q4]=phi[q4]+2*np.pi*u.rad
    bphi=phi-np.pi/2*u.rad
    neg=(bphi < 0)
    if (len(bphi[neg])>0): bphi[neg]=bphi[neg]+2*np.pi*u.rad
    phi_th = bphi - th.to(u.rad)

    c1=(np.cos(inc)*np.cos(rho)-np.sin(inc)*np.sin(rho)*np.sin(phi_th))/(d0.to(u.km)*np.cos(inc))
    cosg=(np.sin(dr)*np.cos(dro)*np.cos(ar-aro)-np.cos(dr)*np.sin(dro))/np.sin(rho)
    sing=(np.cos(dro)*np.sin(ar-aro))/np.sin(rho)
    cosg[np.isnan(cosg)]=0.
    sing[np.isnan(sing)]=-1.

    v2=(1./c1*(-sing*pmw.to(u.rad/u.s) + cosg*pmn.to(u.rad/u.s))).value*u.km/u.s
    v3=(1./c1*(-cosg*pmw.to(u.rad/u.s) - sing*pmn.to(u.rad/u.s))).value*u.km/u.s
    
    a11=np.sin(rho)*np.cos(phi_th)
    a12=np.sin(rho)*np.cos(inc)*np.sin(phi_th) + np.cos(rho)*np.sin(inc)
    a13=np.sin(rho)*np.sin(inc)*np.sin(phi_th) - np.cos(rho)*np.cos(inc)
    a21=np.cos(rho)*np.cos(phi_th)
    a22=np.cos(rho)*np.cos(inc)*np.sin(phi_th) - np.sin(rho)*np.sin(inc)
    a23=np.cos(rho)*np.sin(inc)*np.sin(phi_th) + np.sin(rho)*np.cos(inc)
    a31=-np.sin(phi_th)
    a32=np.cos(inc)*np.cos(phi_th)
    a33=np.sin(inc)*np.cos(phi_th)
    
    #deta=a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31
    deta=a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31)

    i11=1./deta*(a22*a33 - a23*a32)
    i12=1./deta*(a13*a32 - a12*a33)
    i13=1./deta*(a12*a23 - a13*a22)
    i21=1./deta*(a23*a31 - a21*a33)
    i22=1./deta*(a11*a33 - a13*a31)
    i23=1./deta*(a13*a21 - a11*a23)
    i31=1./deta*(a21*a32 - a22*a31)
    i32=1./deta*(a12*a31 - a11*a32)
    i33=1./deta*(a11*a22 - a12*a21)
    
    v1=vhel
    vx=i11*v1+i12*v2+i13*v3
    vy=i21*v1+i22*v2+i23*v3
    vz=i31*v1+i32*v2+i33*v3

    #print(rho.to(u.deg),bphi.to(u.deg),th.to(u.deg))
    
    c15=(d0.to(u.km)*np.sin(rho))/(np.cos(inc)*np.cos(rho)-np.sin(inc)*np.sin(rho)*np.sin(phi_th))
    xx = c15*(np.cos(inc)*np.cos(phi_th))
    yy = c15*np.sin(phi_th)
    ff=(np.cos(inc)*np.cos(rho)-np.sin(inc)*np.sin(rho)*np.sin(phi_th))/ \
        np.sqrt(np.cos(inc)**2*np.cos(phi_th)**2+np.sin(phi_th)**2)
    rr=d0.to(u.km)*np.sin(rho)/ff
    vr = (xx*vx + yy*vy)/rr
    vphi = -(xx*vy - yy*vx)/rr
   
    if (pmwerr is not None) and (pmnerr is not None) and (vherr is not None):
        v2err = np.sqrt(((sing/c1)**2)*(pmwerr.to(u.rad/u.s))**2 + \
                    (cosg/c1)**2*(pmnerr.to(u.rad/u.s))**2).value*u.km/u.s
        v3err = np.sqrt(((cosg/c1)**2)*(pmwerr.to(u.rad/u.s))**2 + \
                    (sing/c1)**2*(pmnerr.to(u.rad/u.s))**2).value*u.km/u.s
        v1err = vherr.to(u.km/u.s)
        vxerr = np.sqrt(i11**2*v1err**2+i12**2*v2err**2+i13**2*v3err**2)
        vyerr = np.sqrt(i21**2*v1err**2+i22**2*v2err**2+i23**2*v3err**2)
        vzerr = np.sqrt(i31**2*v1err**2+i32**2*v2err**2+i33**2*v3err**2)

        vrerr = np.sqrt((xx/rr)**2*vxerr**2 + (yy/rr)**2*vyerr**2)
        vphierr = np.sqrt((yy/rr)**2*vxerr**2 + (xx/rr)**2*vyerr**2)

        if polar:
            return vx,vy,vz,v1,v2,v3,vr,vphi,xx,yy,vxerr,vyerr,vzerr,v1err, \
                        v2err,v3err,vrerr,vphierr
        else:
            return vx,vy,vz,v1,v2,v3,vxerr,vyerr,vzerr,v1err,v2err,v3err

    if polar:
        return vx,vy,vz,v1,v2,v3,vr,vphi,xx,yy # \ #uncomment for debugging
            # ,i11,i12,i13,i21,i22,i23,i31,i32,i33,a11,a12,a13,a21,a22,a23,a31,a32,a33,deta,phi_th,th,bphi
    else:
        return vx,vy,vz,v1,v2,v3

    
# Function to convert 3-D velocity information back to observed quantities
def reproject(ra,dec,vx,vy,vz,params = None):

    try:
        u1=ra.unit
        ra.to(u.deg)
    except:
        ra=ra*u.deg
    try:
        u1=dec.unit
        dec.to(u.deg)
    except:
        dec=dec*u.deg
    try:
        u1=vx.unit
        vx.to(u.km/u.s)
    except:
        vx=vx*u.km/u.s
    try:
        u1=vy.unit
        vy.to(u.km/u.s)
    except:
        vy=vy*u.km/u.s
    try:
        u1=vz.unit
        vz.to(u.km/u.s)
    except:
        vz=vz*u.km/u.s

    if params is None:
        lmcmod=lmcModel()
        aro=lmcmod.aro
        dro=lmcmod.dro
        vsys=lmcmod.vsys
        didt=lmcmod.didt
        th=lmcmod.th
        v0=lmcmod.v0
        inc=lmcmod.inc
        d0=lmcmod.d0
        r0=lmcmod.r0
        s=lmcmod.s
        mun=lmcmod.mun
        muw=lmcmod.muw
        dthdt=lmcmod.dthdt
        vt=np.sqrt(mun**2+muw**2)*d0
        vt=lmcmod.vt
        tht=lmcmod.tht

    else:
        aro = params['aro']*u.rad
        dro = params['dro']*u.rad
        vsys = params['vsys']*u.km/u.s
        didt = params['didt']*u.deg/u.Gyr
        th = params['th']*u.deg
        v0 = params['v0']*u.km/u.s
        inc = params['inc']*u.deg
        d0 = params['d0']*u.pc
        eta = params['eta']
        s = params['s']
        mun = params['mun']*u.mas/u.yr
        muw = params['muw']*u.mas/u.yr
        dthdt = params['dthdt']*u.deg/u.Gyr

        # Derived parameters
        tht = np.arctan(-muw/mun).to(u.deg)
        vt = (np.sqrt(mun**2+muw**2)*d0).to(u.rad*u.km/u.s).value*u.km/u.s
        r0 = eta*d0
    
    ar=ra.to(u.rad)
    dr=dec.to(u.rad)
    ar[ar-aro > np.pi*u.rad] = ar[ar-aro > np.pi*u.rad] - 2*np.pi*u.rad # to set ar-aro between -pi,pi
    ar[ar-aro < -np.pi*u.rad] = ar[ar-aro < -np.pi*u.rad] + 2*np.pi*u.rad # to set ar-aro between -pi,pi

    rho=np.arccos((np.cos(dr)*np.cos(dro)*np.cos(ar-aro)+np.sin(dr)*np.sin(dro)))
    tphi=(np.sin(dr)*np.cos(dro)-np.cos(dr)*np.sin(dro)*np.cos(ar-aro))/(-np.cos(dr)*np.sin(ar-aro))
    tphi[np.isnan(tphi)]=0
    q2=((aro-ar) < 0) & ((dr-dro) > 0)
    q3=((aro-ar) < 0) & ((dr-dro) < 0)
    q4=((aro-ar) > 0) & ((dr-dro) < 0)
    phi=np.arctan(tphi)

    if (len(phi[q2])>0): phi[q2]=phi[q2]+np.pi*u.rad
    if (len(phi[q3])>0): phi[q3]=phi[q3]+np.pi*u.rad
    if (len(phi[q4])>0): phi[q4]=phi[q4]+2*np.pi*u.rad
    bphi=phi-np.pi/2*u.rad
    neg=(bphi < 0)
    if (len(bphi[neg])>0): bphi[neg]=bphi[neg]+2*np.pi*u.rad
    phi_th = bphi - th.to(u.rad)

    a11=np.sin(rho)*np.cos(phi_th)
    a12=np.sin(rho)*np.cos(inc)*np.sin(phi_th)+np.cos(rho)*np.sin(inc)
    a13=np.sin(rho)*np.sin(inc)*np.sin(phi_th)-np.cos(rho)*np.cos(inc)
    a21=np.cos(rho)*np.cos(phi_th)
    a22=np.cos(rho)*np.cos(inc)*np.sin(phi_th)-np.sin(rho)*np.sin(inc)
    a23=np.cos(rho)*np.sin(inc)*np.sin(phi_th)+np.sin(rho)*np.cos(inc)
    a31=-np.sin(phi_th)
    a32=np.cos(inc)*np.cos(phi_th)
    a33=np.sin(inc)*np.cos(phi_th)

    v1=a11*vx+a12*vy+a13*vz
    v2=a21*vx+a22*vy+a23*vz
    v3=a31*vx+a32*vy+a33*vz

    c1=(np.cos(inc)*np.cos(rho)-np.sin(inc)*np.sin(rho)*np.sin(phi_th))/(d0.to(u.km)*np.cos(inc))
    cosg=(np.sin(dr)*np.cos(dro)*np.cos(ar-aro)-np.cos(dr)*np.sin(dro))/np.sin(rho)
    sing=(np.cos(dro)*np.sin(ar-aro))/np.sin(rho)
    cosg[np.isnan(cosg)]=0.
    sing[np.isnan(sing)]=-1.

    muw=(c1*(-v2*sing-v3*cosg)*u.rad).to(u.mas/u.yr)
    mun=(c1*(v2*cosg-v3*sing)*u.rad).to(u.mas/u.yr)    
    
    return v1,muw,mun
    
def addvel3d(lmcTable,params,rakey='RA',deckey='DEC',velkey='VHEL'):

    modvel(lmcTable,params=params,rakey=rakey,deckey=deckey)
    if 'pmra_error' in lmcTable.colnames and 'pmdec_error' in lmcTable.colnames and \
        'VERR' in lmcTable.colnames:
        vx,vy,vz,v1,v2,v3,vr,vphi,xx,yy,vxerr,vyerr,vzerr,v1err,v2err,v3err,vrerr,vphierr = \
            deproject(lmcTable[rakey],lmcTable[deckey],-1*lmcTable['pmra'] - lmcTable['MUWCM'] - \
                     lmcTable['MUWPN'],lmcTable['pmdec'] - lmcTable['MUNCM'] - lmcTable['MUNPN'], \
                     lmcTable[velkey] - (lmcTable['mv1'] + lmcTable['mv2'] + lmcTable['mv3']), \
                     pmwerr=lmcTable['pmra_error'],pmnerr=lmcTable['pmdec_error'],vherr=lmcTable['VERR'], \
                     params=params,polar=True)

        lmcTable['vx'] = vx
        lmcTable['vy'] = vy
        lmcTable['vz'] = vz
        lmcTable['v1'] = v1
        lmcTable['v2'] = v2
        lmcTable['v3'] = v3
        lmcTable['vr'] = vr
        lmcTable['vphi'] = vphi
        lmcTable['xx'] = xx
        lmcTable['yy'] = yy
        lmcTable['vxerr'] = vxerr
        lmcTable['vyerr'] = vyerr
        lmcTable['vzerr'] = vzerr
        lmcTable['v1err'] = v1err
        lmcTable['v2err'] = v2err
        lmcTable['v3err'] = v3err
        lmcTable['vrerr'] = vrerr
        lmcTable['vphierr'] = vphierr
    else:
        vx,vy,vz,v1,v2,v3,vr,vphi,xx,yy = \
                deproject(lmcTable[rakey],lmcTable[deckey],-1*lmcTable['pmra'] - lmcTable['MUWCM'] - \
                     lmcTable['MUWPN'],lmcTable['pmdec'] - lmcTable['MUNCM'] - lmcTable['MUNPN'], \
                     lmcTable[velkey] - (lmcTable['mv1'] + lmcTable['mv2'] + lmcTable['mv3']), \
                     params=params,polar=True)

        lmcTable['vx'] = vx # in-plane velocity along line of nodes
        lmcTable['vy'] = vy # in-plane velocity perpendicular to line of nodes
        lmcTable['vz'] = vz # out-of-plane velocity, +z approximately towards observer
        lmcTable['v1'] = v1 # line-of-sight velocity
        lmcTable['v2'] = v2 # velocity in sky plane along rho direction
        lmcTable['v3'] = v3 # velocity in sky plane along phi direction
        lmcTable['vr'] = vr # in-plane velocity in radial direction
        lmcTable['vphi'] = vphi # in-plane velocity in phi direction
        lmcTable['xx'] = xx # in-plane position along line of nodes
        lmcTable['yy'] = yy # in-plane position perpendicular to line of nodes


def xyz(lmcTable,inc,th,d0,distkey='distance'):
    
    dd = lmcTable[distkey]
    rho = lmcTable['rho']
    phi = lmcTable['bphi']
    xp = dd*np.sin(rho)*np.cos(phi-th)
    yp = dd*(np.sin(rho)*np.cos(inc)*np.sin(phi-th) + np.cos(rho)*np.sin(inc)) - d0*np.sin(inc)
    zp = dd*(np.sin(rho)*np.sin(inc)*np.sin(phi-th) - np.cos(rho)*np.cos(inc)) + d0*np.cos(inc)
    
    return xp,yp,zp

# Function to compute all observed velocities and proper motions given a set of model parameters
# Equations from van der Marel et al. (2002)
def vel3d_resid(params, x, data=None, errors=None, concat=True):

    # Observed positions
    # x: (ra,dec)
    #ra=lmcTable['RA']*u.deg
    #dec=lmcTable['DEC']*u.deg

    # Model parameters
    aro = params['aro']*u.rad
    dro = params['dro']*u.rad
    vsys = params['vsys']*u.km/u.s
    didt = params['didt']*u.deg/u.Gyr
    th = params['th']*u.deg
    v0 = params['v0']*u.km/u.s
    inc = params['inc']*u.deg
    d0 = params['d0']*u.pc
    eta = params['eta']
    s = params['s']
    mun = params['mun']*u.mas/u.yr
    muw = params['muw']*u.mas/u.yr
    dthdt = params['dthdt']*u.deg/u.Gyr

    # Derived parameters
    tht = np.arctan(-muw/mun).to(u.deg)
    vt = (np.sqrt(mun**2+muw**2)*d0).to(u.rad*u.km/u.s).value*u.km/u.s
    r0 = eta*d0
    
    ra = x[0,:]*u.deg
    dec = x[1,:]*u.deg
    ar = ra.to(u.rad)
    dr = dec.to(u.rad)
    ar[ar-aro > np.pi*u.rad] = ar[ar-aro > np.pi*u.rad] - 2*np.pi*u.rad # to set ar-aro between -pi,pi
    ar[ar-aro < -np.pi*u.rad] = ar[ar-aro < -np.pi*u.rad] + 2*np.pi*u.rad # to set ar-aro between -pi,pi

    muc=-muw*np.sin(th) + mun*np.cos(th)
    mus=-muw*np.cos(th) - mun*np.sin(th)
    wts=d0.to(u.km)*(didt.to(u.rad/u.s) + mus.to(u.rad/u.s))
    wts=wts.value*u.km/u.s
    vtc=d0.to(u.km)*muc.to(u.rad/u.s)
    vtc=vtc.value*u.km/u.s

    rho=np.arccos((np.cos(dr)*np.cos(dro)*np.cos(ar-aro)+np.sin(dr)*np.sin(dro)))
    tphi=(np.sin(dr)*np.cos(dro)-np.cos(dr)*np.sin(dro)*np.cos(ar-aro))/(-np.cos(dr)*np.sin(ar-aro))
    tphi[np.isnan(tphi)]=0
    q2=((aro-ar) < 0) & ((dr-dro) > 0)
    q3=((aro-ar) < 0) & ((dr-dro) < 0)
    q4=((aro-ar) > 0) & ((dr-dro) < 0)
    phi=np.arctan(tphi)

    if (len(phi[q2])>0): phi[q2]=phi[q2]+np.pi*u.rad
    if (len(phi[q3])>0): phi[q3]=phi[q3]+np.pi*u.rad
    if (len(phi[q4])>0): phi[q4]=phi[q4]+2*np.pi*u.rad
    bphi=phi-np.pi/2*u.rad
    neg=(bphi < 0)
    if (len(bphi[neg])>0): bphi[neg]=bphi[neg]+2*np.pi*u.rad

    ff=(np.cos(inc)*np.cos(rho)-np.sin(inc)*np.sin(rho)*np.sin(bphi-th))/np.sqrt(np.cos(inc)**2*np.cos(bphi-th)**2+np.sin(bphi-th)**2)
    rpri=d0*np.sin(rho)/ff
    vr=(v0/r0*rpri)
    vr=np.clip(vr,0,v0)

    # model line of sight velocities
    mv1=vsys*np.cos(rho) # contribution from center of mass motion
    mv2=wts*np.sin(rho)*np.sin(bphi-th) # contribution from transverse motion, component 1
    mv3=(vtc*np.sin(rho))*np.cos(bphi-th) # contribution from transverse motion, component 2
    vrfac=(-s*ff*np.sin(inc))*np.cos(bphi-th)
    mv4=(-s*ff*vr*np.sin(inc))*np.cos(bphi-th) # contribution from internal rotation
    modv=mv1+mv2+mv3+mv4 # total line of sight velocity of model

    # model proper motions
    # broken up by contribution from different sources; 
    # labels 1,2,and 3 refer to x,y,z directions

    # Center-of-mass motion contribution
    mv2cm=vt*np.cos(rho)*np.cos(bphi-tht)-vsys*np.sin(rho)
    mv3cm=-vt*np.sin(bphi-tht)

    div1=np.cos(inc)*np.cos(rho)-np.sin(inc)*np.sin(rho)*np.sin(bphi-th)

    # Contribution from precession and nutation
    mv2pn=d0.to(u.km)*np.sin(rho)/div1*didt.to(u.rad/u.s)*np.sin(bphi-th)*(-np.cos(inc)*np.sin(rho)-np.sin(inc)*np.cos(rho)*np.sin(bphi-th))
    mv2pn=mv2pn.value*u.km/u.s
    mv3pn=d0.to(u.km)*np.sin(rho)/div1*(didt.to(u.rad/u.s)*np.sin(bphi-th)*(-np.sin(inc)*np.cos(bphi-th))+dthdt.to(u.rad/u.s)*np.cos(inc))
    mv3pn=mv3pn.value*u.km/u.s

    # Contribution from internal rotation
    ff2=(np.cos(inc)*np.sin(rho)+np.sin(inc)*np.cos(rho)*np.sin(bphi-th))/np.sqrt(np.cos(inc)**2*np.cos(bphi-th)**2+np.sin(bphi-th)**2)
    mv2int=s*ff2*vr*np.sin(inc)*np.cos(bphi-th)

    ff3=(np.cos(inc)**2*np.cos(bphi-th)**2+np.sin(bphi-th)**2)/np.sqrt(np.cos(inc)**2*np.cos(bphi-th)**2+np.sin(bphi-th)**2)
    mv3int=-s*vr*ff3

    cosg=(np.sin(dr)*np.cos(dro)*np.cos(ar-aro)-np.cos(dr)*np.sin(dro))/np.sin(rho)
    sing=(np.cos(dro)*np.sin(ar-aro))/np.sin(rho)
    cosg[np.isnan(cosg)]=0.
    sing[np.isnan(sing)]=-1.

    # Total transverse (proper motion) velocities
    mv2pm=mv2cm+mv2pn+mv2int
    mv3pm=mv3cm+mv3pn+mv3int

    # Transform to observed proper motions in W and N directions
    c1=(np.cos(inc)*np.cos(rho)-np.sin(inc)*np.sin(rho)*np.sin(bphi-th))/(d0.to(u.km)*np.cos(inc))
    muwmod=(c1*(-mv2pm*sing-mv3pm*cosg)*u.rad).to(u.mas/u.yr)
    munmod=(c1*(mv2pm*cosg-mv3pm*sing)*u.rad).to(u.mas/u.yr)

    muwcm=(c1*(-mv2cm*sing-mv3cm*cosg)*u.rad).to(u.mas/u.yr)
    muncm=(c1*(mv2cm*cosg-mv3cm*sing)*u.rad).to(u.mas/u.yr)

    muwpn=(c1*(-mv2pn*sing-mv3pn*cosg)*u.rad).to(u.mas/u.yr)
    munpn=(c1*(mv2pn*cosg-mv3pn*sing)*u.rad).to(u.mas/u.yr)

    muwint=(c1*(-mv2int*sing-mv3int*cosg)*u.rad).to(u.mas/u.yr)
    munint=(c1*(mv2int*cosg-mv3int*sing)*u.rad).to(u.mas/u.yr)

    # Store model proper motions and line of sight velocities
    # (including separate contributions) in data table
    model1 = modv.value # line of sight
    model2 = -muwmod.value # p.m. E-W, +E, -W
    model3 = munmod.value # p.m. N-S, +N, -S
    
    model = np.concatenate([model1,model2,model3])

    if concat:
        if data is None:
            return model
        if errors is None:
            return model - data
        return (model - data) / errors
    else:
        return model1,model2,model3
        

        
def vhplot_grid(lmcTable,params,velkey='VHEL',figsize=[20,30],xlim=(110,50),ylim=(-80,-60),plot_center=True):

    aro=params['aro']*u.rad
    dro=params['dro']*u.rad
    lmcCenter=Table(names=['RA','DEC'])
    lmcCenter.add_row([aro.to(u.deg).value,dro.to(u.deg).value])
    modvel(lmcCenter,params=params)
    
    fig = plt.figure(figsize=figsize)

    # model components 1,2,3 (space motion)
    ax1 = fig.add_subplot(321)
    norm1 = matplotlib.colors.Normalize(vmin=lmcCenter['mvtot']-200, vmax=lmcCenter['mvtot']+200, clip=True)
    col1 = lmcTable['mv1'] + lmcTable['mv2'] + lmcTable['mv3']
    ax1.scatter(lmcTable['RA'],lmcTable['DEC'],c=col1,marker='.',cmap='seismic',alpha=1,norm=norm1)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_title('Model space motion')
    if plot_center:
        ax1.scatter(lmcCenter['RA'],lmcCenter['DEC'],marker='x',s=50)
        
    # model component 4 (internal motion)
    ax2 = fig.add_subplot(322)
    norm2 = matplotlib.colors.Normalize(vmin=-200, vmax=200, clip=True)
    col2 = lmcTable['mv4']
    ax2.scatter(lmcTable['RA'],lmcTable['DEC'],c=col2,marker='.',cmap='seismic',alpha=1,norm=norm2)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_title('Model internal motion')
    if plot_center:
        ax2.scatter(lmcCenter['RA'],lmcCenter['DEC'],marker='x',s=50)

    
    # total model
    ax3 = fig.add_subplot(323)
    norm3 = norm1
    col3 = lmcTable['mvtot']
    ax3.scatter(lmcTable['RA'],lmcTable['DEC'],c=col3,marker='.',cmap='seismic',alpha=1,norm=norm3)
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.set_title('Total model')
    if plot_center:
        ax3.scatter(lmcCenter['RA'],lmcCenter['DEC'],marker='x',s=50)


    # total obs
    ax4 = fig.add_subplot(324)
    norm4 = norm1
    col4 = lmcTable[velkey]
    ax4.scatter(lmcTable['RA'],lmcTable['DEC'],c=col4,marker='.',cmap='seismic',alpha=1,norm=norm4)
    ax4.set_xlim(xlim)
    ax4.set_ylim(ylim)
    ax4.set_title('Total observed')
    if plot_center:
        ax4.scatter(lmcCenter['RA'],lmcCenter['DEC'],marker='x',s=50)


    # residual
    ax5 = fig.add_subplot(325)
    norm5 = norm2
    col5 = lmcTable[velkey] - lmcTable['mvtot']
    ax5.scatter(lmcTable['RA'],lmcTable['DEC'],c=col5,marker='.',cmap='seismic',alpha=1,norm=norm5)
    ax5.set_xlim(xlim)
    ax5.set_ylim(ylim)
    ax5.set_title('Data - Model')
    if plot_center:
        ax5.scatter(lmcCenter['RA'],lmcCenter['DEC'],marker='x',s=50)


    # obs - space motion
    ax6 = fig.add_subplot(326)
    norm6 = norm2
    col6 = lmcTable[velkey] - (lmcTable['mv1'] + lmcTable['mv2'] + lmcTable['mv3'])
    ax6.scatter(lmcTable['RA'],lmcTable['DEC'],c=col6,marker='.',cmap='seismic',alpha=1,norm=norm6)
    ax6.set_xlim(xlim)
    ax6.set_ylim(ylim)
    ax6.set_title('Data - Model space motion')
    if plot_center:
        ax6.scatter(lmcCenter['RA'],lmcCenter['DEC'],marker='x',s=50)


# Proper motion plotting function
def arrowplot(x,y,dx,dy,x0,y0,dx0,dy0,figsize=[10,10],xlim=(110,50),ylim=(-80,-60),
              alpha1=1,alpha2=1,scale=1,width=1,plot_center=True,fig=None,ax=None):

    figreturn = False
    if fig is None:
        fig = plt.figure(figsize=figsize)
        figreturn = True
    if ax is None:
        ax = fig.add_subplot(111)
        
    ax.scatter(x,y,marker='.',alpha=alpha1)
    ax.quiver(x,y,dx,dy,color='r',alpha=alpha2,angles='xy',scale_units='xy',scale=scale,width=width)
    if plot_center:
        ax.arrow(x0,y0,dx0,dy0,color='k',width=0.05)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('RA',fontsize=14)
    ax.set_ylabel('Dec',fontsize=14)
    
    if figreturn:
        return ax,fig
    

def pmplot(table,params,figsize=(10,10),xlim=(110,50),ylim=(-80,-60),
              alpha1=1,alpha2=1,scale=1,width=1,plot_center=True,fig=None,ax1=None,ax2=None):

    figreturn = False
    if fig is None:
        fig = plt.figure(figsize=figsize)
        figreturn = True
    if (ax1 is None) | (ax2 is None):
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
    
    xp = table['RA']
    yp = table['DEC']
    xp0 = params['aro'].value*u.rad.to(u.deg)
    yp0 = params['dro'].value*u.rad.to(u.deg)
    dxp0 = -1*params['muw'].value/np.cos(yp0*u.deg)
    dyp0 = params['mun'].value/np.cos(yp0*u.deg)
    dxp2 = (table['pmra'] - -1*table['MUWMOD'])/np.cos(yp*u.deg)
    dyp2 = table['pmdec'] - table['MUNMOD']
    dxp1 = (table['pmra'] - -1*table['MUWCM'] - -1*table['MUWPN'])/np.cos(yp*u.deg)
    dyp1 = table['pmdec'] - table['MUNCM'] - table['MUNPN']

    arrowplot(xp,yp,dxp1,dyp1,xp0,yp0,dxp0,dyp0,alpha1=alpha1,alpha2=alpha2,fig=fig,ax=ax1,scale=scale,width=width)
    ax1.set_title('Data - COM motion')
    arrowplot(xp,yp,dxp2,dyp2,xp0,yp0,dxp0,dyp0,alpha1=alpha1,alpha2=alpha2,fig=fig,ax=ax2,scale=scale,width=width)
    ax2.set_title('Data - Model')

    if figreturn:
        return ax1,ax2,fig
    
def pmplot1(table,params,figsize=(10,10),xlim=(110,50),ylim=(-80,-60),
              alpha1=1,alpha2=1,scale=1,width=1,plot_center=True,fig=None,ax1=None):

    figreturn = False
    if fig is None:
        fig = plt.figure(figsize=figsize)
        figreturn = True
    if (ax1 is None):
        ax1 = fig.add_subplot(111)
    
    xp = table['RA']
    yp = table['DEC']
    xp0 = params['aro'].value*u.rad.to(u.deg)
    yp0 = params['dro'].value*u.rad.to(u.deg)
    dxp0 = -1*params['muw'].value/np.cos(yp0*u.deg)
    dyp0 = params['mun'].value/np.cos(yp0*u.deg)
    dxp2 = (table['pmra'] - -1*table['MUWMOD'])/np.cos(yp*u.deg)
    dyp2 = table['pmdec'] - table['MUNMOD']
    dxp1 = (table['pmra'] - -1*table['MUWCM'] - -1*table['MUWPN'])/np.cos(yp*u.deg)
    dyp1 = table['pmdec'] - table['MUNCM'] - table['MUNPN']

    arrowplot(xp,yp,dxp2,dyp2,xp0,yp0,dxp0,dyp0,alpha1=alpha1,alpha2=alpha2,fig=fig,ax=ax1,scale=scale,width=width)
    ax1.set_title('Data - Model')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    
    if figreturn:
        return ax1,fig
    
# Line-of-sight velocity plotting function
def velplot(dz,figsize=[10,10],xlim=(110,50),ylim=(-80,-60),
              vel1=-100,vel2=100):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    im = ax.imshow(dz,cmap='seismic',norm=matplotlib.colors.Normalize(vmin=vel1,vmax=vel2),
                  extent=[xlim[0],xlim[1],ylim[0],ylim[1]],aspect='auto')
    #ax.scatter(x,y,marker='o',c=dz,cmap='seismic',alpha=alpha,s=s,norm=matplotlib.colors.Normalize(vmin=vel1,vmax=vel2))
    #ax.set_xlim(xlim)
    #ax.set_ylim(ylim)
    ax.set_xlabel('RA',fontsize=14)
    ax.set_ylabel('Dec',fontsize=14)
    
    return ax,fig


