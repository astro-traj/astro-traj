# -*- coding: utf-8 -*-
# Copyright (C) Scott Coughlin (2017)
#
# This file is part of astro-traj.
#
# astro-traj is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# astro-traj is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with astro-traj.  If not, see <http://www.gnu.org/licenses/>.

"""
Places system described by Mhe, M2, Apre, epre and position r(R,galphi,galcosth) in galaxy model gal
Applies SNkick Vkick and mass loss Mhe-Mns to obtain Apost, epost, and SN-imparted systemic velocity V    
"""

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as C
from scipy.integrate import ode
from scipy.stats import maxwell
from scipy.stats import rv_continuous
from scipy.integrate import quad

__author__ = ['Chase Kimball <charles.kimball@ligo.org>', 'Michael Zevin <michael.zevin@ligo.org>']
__credits__ = 'Scott Coughlin <scott.coughlin@ligo.org>'
__all__ = ['System']

class System:
    """
    Places system described by Mhe, M2, Apre, epre and position r(R,galphi,galcosth) in galaxy model gal

    Applies SNkick Vkick and mass loss Mhe-Mns to obtain Apost, epost, and SN-imparted systemic velocity V
    
    """
    def __init__(self, gal, R, Mns, M2, Mhe, Apre, epre, d, Vkick, sys_flag=None, galphi=None, galcosth=None, omega=None, phi=None, costh=None):
        """ 
        #Masses in Msun, Apre in Rsun, Vkick in km/s, R in kpc
        #galphi,galcosth,omega, phi, costh (position, initial velocity, and kick angles) sampled randomly, unless specified (>-1)
        #galphi, galcosth correspond to azimuthal and polar angles -- respectively --  in the galactic frame
        #phi, costh are defined in comments of SN:
        #   theta: angle between preSN He core velocity relative to M2 (i.e. the positive y axis) and the kick velocity
        #   phi: angle between Z axis and projection of kick onto X-Z plane
        #omega: angle between the galactic velocity corresponding to a circular orbit in the r-z plane and
        #the actual galactic velocity preSN corresponding to a circular orbit
        
        """
    
        # Convert inputs to SI
        Mhe = Mhe*u.M_sun.to(u.kg)
        M2 = M2*u.M_sun.to(u.kg)
        Mns = Mns*u.M_sun.to(u.kg)
        Apre = Apre*u.R_sun.to(u.m)
        Vkick = Vkick*u.km.to(u.m)
        R = R*u.kpc.to(u.m)
        d = d*u.Mpc.to(u.m)

        self.sys_flag = sys_flag
        
        if galphi: self.galphi = galphi
        else: self.galphi = np.random.uniform(0,2*np.pi)

        if galcosth: self.galcosth = galcosth
        else: self.galcosth = np.random.uniform(-1,1)

        if phi: self.phi = phi
        else: self.phi = np.random.uniform(0,2*np.pi)

        if costh: self.costh = costh
        else: self.costh = np.random.uniform(-1,1)

        if omega: self.omega = omega
        else: self.omega = np.random.uniform(0,2*np.pi)

        self.Mhe, self.M2, self.Mns, self.Apre, self.epre, self.Vkick, self.gal, self.R, self.d = Mhe, M2, Mns, Apre, epre, Vkick, gal, R, d
        self.Vdot = gal.Vdot

        # Get projection of R in the x-y plane to save later into output file
        x_R = self.R*np.sin(np.arccos(self.galcosth))*np.cos(self.galphi)
        y_R = self.R*np.sin(np.arccos(self.galcosth))*np.sin(self.galphi)
        z_R = self.R*self.galcosth
        self.R_proj = np.sqrt(x_R**2 + y_R**2)

    def SN(self):
        """
        
        Mhe lies on origin moving in direction of positive y axis, M2 on negative X axis, Z completes right-handed coordinate system
        
        theta: angle between preSN He core velocity relative to M2 (i.e. the positive y axis) and the kick velocity
        phi: angle between Z axis and projection of kick onto X-Z plane
        
        Vr is velocity of preSN He core relative to M2, directed along the positive y axis

        Vkick is kick velocity with components Vkx, Vky, Vkz in the above coordinate system
        V_sys is the resulting center of mass velocity of the system IN THE TRANSLATED COM FRAME, imparted by the SN

        Paper reference:

        Kalogera 1996: http://iopscience.iop.org/article/10.1086/177974/meta
            We use Eq 1, 3, 4, and 34: giving Vr, Apost, epost, and (Vsx,Vsy,Vsz) respectively
            Also see Fig 1 in that paper for coordinate system

        
        """

        self.flag=0      # set standard flag        

        G = C.G.value
        Mhe, M2, Mns, Apre, Vkick, costh, phi = self.Mhe, self.M2, self.Mns, self.Apre, self.Vkick, self.costh, self.phi


        sinth = np.sqrt(1-(costh**2))
        #Mhe lies on origin moving in direction of positive y axis, M2 on negative X axis, Z completes right-handed coordinate system
        #See Fig 1 in Kalogera 1996

        # theta: angle between preSN He core velocity relative to M2 (i.e. the positive y axis) and the kick velocity
        # phi: angle between Z axis and projection of kick onto X-Z plane
        Vkx = Vkick*sinth*np.sin(phi)
        Vky = Vkick*costh
        Vkz = Vkick*sinth*np.cos(phi)
        if self.sys_flag == 'radial_simple' or self.sys_flag == 'tangential' or self.sys_flag == 'radial_simple2' or self.sys_flag == 'tangential2':
            Vkx,Vky,Vkz=0,-Vkick,0
        #Eq 1, Kalogera 1996
        Vr = np.sqrt(G*(Mhe+M2)/Apre)
        Mtot=Mns+M2

        #Eqs 3 and 4, Kalogera 1996
        Apost = ((2.0/Apre) - (((Vkick**2)+(Vr**2)+(2*Vky*Vr))/(G*Mtot)))**-1
        x = ((Vkz**2)+(Vky**2)+(Vr**2)+(2*Vky*Vr))*(Apre**2)/(G*Mtot*Apost)
        epost = np.sqrt(1-x)
        # Eq 34, Kalogera 1996
        VSx = Mns*Vkx/Mtot
        VSy = (1.0/Mtot)*((Mns*Vky)-((Mhe-Mns)*M2*Vr/(Mhe+M2)))
        VSz = Mns*Vkz/Mtot
        V_sys = np.sqrt((VSx**2)+(VSy**2)+(VSz**2))

        self.Apost, self.epost, self.VSx, self.VSy, self.VSz, self.V_sys, self.Vr = Apost, epost, VSx, VSy, VSz, V_sys, Vr
        
        def SNCheck(self):
            """
            Paper References:

            Willems et al 2002: http://iopscience.iop.org/article/10.1086/429557/meta
                We use eq 21, 22, 23, 24, 25, 26 for checks of SN survival

            Kalogera and Lorimer 2000: http://iopscience.iop.org/article/10.1086/308417/meta

            
            V_He;preSN is the same variable as V_r from Kalogera 1996
            
            """
            Mhe, M2, Mns, Apre, Apost, epost, Vr, Vkick = self.Mhe, self.M2, self.Mns, self.Apre, self.Apost, self.epost, self.Vr, self.Vkick
            #Equation numbers and quotes in comments correspond to Willems et al. 2002 paper on J1655.
            Mtot_pre = Mhe + M2
            Mtot_post = Mns + M2

            # SNflag1: eq 21 (with typo fixed). Continuity demands Post SN orbit must pass through preSN positions.
            #from Flannery & Van Heuvel 1975                                                             

            self.SNflag1 = (1-epost <= Apre/Apost) and (Apre/Apost <= 1+epost)


            # SNflag2: Equations 22 & 23. "Lower and upper limits on amount of orbital contraction or expansion that can take place                                
            #for a given amount of mass loss and a given magnitude of the kick velocity (see, e.g., Kalogera & Lorimer 2000)"                            

            self.SNflag2 = (Apre/Apost < 2-((Mtot_pre/Mtot_post)*((Vkick/Vr)-1)**2)) and (Apre/Apost > 2-((Mtot_pre/Mtot_post)*((Vkick/Vr)+1)**2))

            #SNflag3: Equations 24 and 25."The magnitude of the kick velocity imparted to the BH at birth is restricted to the
            #range determined by (Brandt & Podsiadlowski 1995; Kalogera & Lorimer 2000)
            #the first inequality expresses the requirement that the binary must remain bound after the SN explosion,
            #while the second inequality yields the minimum kick velocity required to keep the system bound if more than
            #half of the total system mass is lost in the explosion.

            self.SNflag3 = (Vkick/Vr < 1 + np.sqrt(2*Mtot_post/Mtot_pre)) and ((Mtot_post/Mtot_pre > 0.5) or (Vkick/Vr>1 - np.sqrt(2*Mtot_post/Mtot_pre)))

            #SNflag4: Eq 26 "An upper limit on the mass of the BH progenitor can be derived from the condition that the
            #azimuthal direction of the kick is real (Fryer & Kalogera 1997)"
            if epost>1: self.SNflag4 = False
            else:
                kvar=2*(Apost/Apre)-(((Vkick**2)*Apost/(G*Mtot_post))+1)

                tmp1 = kvar**2 * Mtot_post * (Apre/Apost)
                tmp2 = 2 * (Apost/Apre)**2 * (1-epost**2) - kvar
                tmp3 = - 2 * (Apost/Apre) * np.sqrt(1-epost**2) * np.sqrt((Apost/Apre)**2 * (1-epost**2) - kvar)
                prgmax = -M2 + tmp1 / (tmp2 + tmp3)

                self.SNflag4 = Mhe <= prgmax
            # FIX ME: additionally, Kalogera 1996 mentions requirement that NS stars don't collide
            # Apost*(1-epost)> Rns1+Rns2    (eq 16 in that paper)
            # Is there analytic expression for NS radius?

            self.SNflags = [self.SNflag1, self.SNflag2, self.SNflag3, self.SNflag4]

            # check if the supernova is valid and doesn't disrupt the system
            if (np.asarray(self.SNflags) == False).any():
                self.flag = 3

        SNCheck(self)




    def getVcirc(self,X,Y,Z): #velocity of circular orbit in galactic potential at R
        """ 
        Calculate circular velocity at X,Y,Z given potential. From mv2/r = -grad_r(U)
        FIXME: Will have to change for spiral potential, as circular velocity is assumed to be within the disk
        """
        Vdot = self.Vdot
        
        COORD=[0,0,0,X,Y,Z]
        r=np.sqrt((X**2)+(Y**2)+(Z**2))

        gradUx,gradUy,gradUz = Vdot(0,COORD)[:3]
        gradU = np.sqrt((gradUx**2)+(gradUy**2)+(gradUz**2))

        vcirc = np.sqrt(r*gradU)

        return vcirc

    def setXYZ_0(self):
        """ 
        Convert from spherical inputs to Cartesian coordinates
        """
        R = self.R
        galcosth = self.galcosth
        galphi = self.galphi
        galsinth = np.sqrt(1-(galcosth**2))


        self.X0 = R*galsinth*np.cos(galphi)
        self.Y0 = R*galsinth*np.sin(galphi)
        self.Z0 = R*galcosth

    def setVxyz_0(self):
        """ 
        Here, vphi and vcosth are as galphi and galcosth, and give random direction for V_sys postSN

        Initially, preSN circular trajectory Vp is the velocity vector of magnitude getVcirc,
        tangential to the sphere of radius R, in the R-Z plane

        We get the specific circular trajectory that we want by rotating this through angle omega while staying
        tangential to the sphere, giving Vp_rot (Vp_rot = Vp cos(omega) + (k x Vp) sin(omega)
        where k is unit vector r/R where r=(x,y,z)

        Specify flag=circ_test to set V0 without adding the SN-imparted velocity
        Specify flag=vkick_test to set the initial galactic velocity to 0, so that the velocity of the system is due solely to the supernova


        """
        if self.sys_flag:
            if self.sys_flag not in ['circ_test','vkick_test','radial_iso','radial_x','radial_simple','tangential','radial_simple2','tangential2']:
                raise ValueError("Unspecified flag '%s'" % self.sys_flag)

        X0,Y0,Z0 = self.X0, self.Y0, self.Z0
        R = self.R

        galphi, galcosth = self.galphi, self.galcosth
        galsinth = np.sqrt(1-(galcosth**2))
        galth = np.arccos(galcosth)

        omega = self.omega #orientation of orbit on R-sphere (angle between Vorb and R-Z plane)

        if self.sys_flag=='circ_test': V_sys = 0 #For checking that initial conditions correspond to circular galactic orbits
        else: V_sys = self.V_sys #Velocity imparted by SN

        vphi = np.random.uniform(0,2*np.pi) #
        vcosth = np.random.uniform(-1,1)    # Choose random direction for system velocity
        vsinth = np.sqrt(1-(vcosth**2))    # equivalent to choosing random orientation preSN
        Vtot = self.getVcirc(X0,Y0,Z0)
        

        

        if self.sys_flag=='vkick_test': Vtot = 0 #For checking that initial conditions correspond to circular galactic orbits
        if self.sys_flag=='radial_iso':
            Vtot = 0
            vphi, vcosth, vsinth =  galphi,galcosth,galsinth

        vsys = np.array([V_sys*vsinth*np.cos(vphi),V_sys*vsinth*np.sin(vphi),V_sys*vcosth])

        vpx = Vtot * np.sin(galth-(np.pi/2))*np.cos(galphi)
        vpy = Vtot * np.sin(galth-(np.pi/2))*np.sin(galphi)
        vpz = Vtot * np.cos(galth-(np.pi/2))

        Vp = np.array([vpx,vpy,vpz]) #velocity of circular orbit in R-Z plane
        k = np.array([X0,Y0,Z0])/R #unit vector in direction of R

        #Rotate by omega while keeping perpendicular to R
        Vp_rot = (Vp*np.cos(omega)) + (np.cross(k,Vp)*np.sin(omega))
        Vp_rot_tot = np.sqrt((Vp[0]**2)+(Vp[1]**2)+(Vp[2]**2))
        if self.sys_flag == 'tangential' or self.sys_flag== 'tangential2':
            vsys = [V_sys*Vp_rot[0]/Vp_rot_tot,V_sys*Vp_rot[1]/Vp_rot_tot,V_sys*Vp_rot[2]/Vp_rot_tot]
        Vx0,Vy0,Vz0 = Vp_rot + vsys
        if self.sys_flag =='radial_x':
            Vtot = 0
            Vp_rot = np.array([0,0,0])
            self.X0,self.Y0,self.Z0 = self.R,0.,0.
            Vx0,Vy0,Vz0 = V_sys,0.,0.
            self.galphi,self.galcosth = 0.,0.
            
            
            
        self.Vxcirc0, self.Vycirc0, self.Vzcirc0 = Vp_rot
        self.Vtot = Vtot

        #Add velocity imparted by SN
        self.Vx0, self.Vy0, self.Vz0 = Vx0,Vy0,Vz0
        self.vsys=vsys
        self.vphi=vphi
        self.vcosth=vcosth
        self.vsinth=vsinth

    def setTmerge(self, Tmin=0.0, Tmax=10.0): #NOTE we should check that this matches up with Maggiori equations
        """ 
        Calculate the inspiral time for the binary after the supernova using formulae from `Peters 1964 <https://journals.aps.org/pr/abstract/10.1103/PhysRev.136.B1224>`_
        """
        m1=self.Mns; m2=self.M2
        G = C.G.value; c = C.c.value

        # useful definition for following equations:
        beta = (64./5)*(G**3)*m1*m2*(m1+m2) / (c**5)

        # function for defining scale factor based on initial conditions
        def c_0(a0,e0):
            return (a0*(1-e0**2)) * (e0**(-12./19)) * (1+(121./304)*e0**2)**(-870./2299)
        c0 = c_0(self.Apost,self.epost)

        # now we integrate Eq. 5.14 for Peters 1964
        def integrand(e):
            return e**(29./19) * (1+(121./304)*e**2)**(1181./2299) / (1-e**2)**(3./2)
        ef = 0.0   # assume binary circularizes by the time it merges
        Tmerge = (12./19)*((c0**4)/beta)*quad(integrand, ef, self.epost)[0]
        self.Tmerge = Tmerge

        # see if binary inspiral time is longer than threshold (10 Gyr) or shorter than minimum time (0 for now)
        Tmin = Tmin*1e9         # years
        Tmax = Tmax*1e9         # years
        if (Tmerge > Tmax * u.year.to(u.s) or Tmerge < Tmin * u.year.to(u.s)):
            self.flag=2   # binary does not meet inspiral time requirements


    def doMotion(self, backend='dopri5', NSTEPS=1e13, MAX_STEP=u.year.to(u.s)*1e6, RTOL=1e-11):
        """ 
        Second order equation ma=-grad(U) converted to 2 sets of first order equations, with
        e.g. x1 = x
             x2 = vx

             x1dot = vx = x2
             x2dot = ax = -grad_x(U)/m
        """
        X0,Y0,Z0 = self.X0, self.Y0, self.Z0
        Vx0,Vy0,Vz0 = self.Vx0, self.Vy0, self.Vz0
        RR = np.array([Vx0,Vy0,Vz0,X0,Y0,Z0])
        Tmerge=self.Tmerge
        Vdot = self.Vdot
        self.RR=RR
        sol = []
        solver=ode(Vdot).set_integrator(backend,nsteps=NSTEPS,max_step=MAX_STEP,rtol=RTOL)

        def solout(t,y):
            """ 
            function for saving integration results to sol[]
            """
            temp=list(y)
            temp.append(t)
            sol.append(temp)


        solver.set_solout(solout)
        solver.set_initial_value(RR,0)
        solver.integrate(Tmerge)
        sol=np.array(sol)
        self.Vx = sol[:,0]
        self.Vy = sol[:,1]
        self.Vz = sol[:,2]

        self.X = sol[:,3]
        self.Y = sol[:,4]
        self.Z = sol[:,5]

        self.t = sol[:,6]


    def check_success(self, offset, uncer=0.5):
        """ 
        # uncertainty in offset is 0.5 kpc by default
        # assume that the observer is looking down the z-axis (so the offset will be the projection of the binary on the x-y plane)
        """
        offset = offset*u.kpc.to(u.m)
        uncer = uncer*u.kpc.to(u.m)
        Rmerge_proj = np.sqrt(self.X[-1]**2 + self.Y[-1]**2)

        self.Rmerge_proj = Rmerge_proj
        self.Rmerge = np.sqrt(self.X[-1]**2 + self.Y[-1]**2 + self.Z[-1]**2)
        self.Vfinal = np.sqrt(self.Vx[-1]**2 + self.Vy[-1]**2 + self.Vz[-1]**2)

        if (offset-uncer < Rmerge_proj < offset+uncer):
            self.flag = 1      # successful binary!
            print 'GW analog produced! R_SN:%f R_Merge:%f R_Merge_proj:%f, Vkick:%f Mhe:%f' % \
                        (self.R/C.kpc.value, self.Rmerge/C.kpc.value, self.Rmerge_proj/C.kpc.value, self.Vkick/1000, self.Mhe/C.M_sun.value)



    def energy_check(self, E_thresh = 1e-3):
        """ 
        Compare total energy of first and last steps to ensure conservation
        Ek, Ep are kinetic and potential energy

        """
        Mhalo = self.gal.Mhalo
        Mbulge = self.gal.Mbulge
        abulge = self.gal.abulge
        Mns, M2 = self.Mns, self.M2
        G = C.G.value
        Ubulge = self.gal.Ubulge
        Uhalo = self.gal.Uhalo

        ri = np.sqrt(self.X[0]**2 + self.Y[0]**2 + self.Z[0]**2)
        rf = np.sqrt(self.X[-1]**2 + self.Y[-1]**2 + self.Z[-1]**2)


        Eki = 0.5 * (Mns + M2) * (self.Vx[0]**2 + self.Vy[0]**2 + self.Vz[0]**2)
        Epi = (Mns+M2)*(Uhalo(ri)+Ubulge(ri))  #Uhalo and Ubulge are really Uhalo/Msys and Ubulge/Msys

        Ei = Eki+Epi

        Ekf = 0.5 * (Mns + M2) * (self.Vx[-1]**2 + self.Vy[-1]**2 + self.Vz[-1]**2)
        Epf = (Mns+M2)*(Uhalo(rf)+Ubulge(rf))  #Uhalo and Ubulge are really Uhalo/Msys and Ubulge/Msys

        Ef = Ekf+Epf


        self.Ei, self.Ef = Ei, Ef
        self.dEfrac = (Ei-Ef)/Ei

        if np.abs((Ei-Ef)/Ei) > E_thresh:
            self.flag=4 # Energy not conserved!!!
        

    def write_data(self):
        """
        # [M2, Mns, Mhe, Apre, Apost, epre, epost, d, R, galcosth, galphi, Vkick, Tmerge, Rmerge, Rmerge_proj, Vfinal, flag]
        # write things in reasonable units (e.g., Msun, kpc, km/s ...)
        """
        # in case we wish to save data from other flagged binaries, fill in the blank information with nans
        if self.flag == 3:
            self.vphi = np.nan
            self.vcosth = np.nan
            self.Tmerge = np.nan
            self.Rmerge = np.nan
            self.Rmerge_proj = np.nan
            self.Vfinal = np.nan
        if self.flag == 2:
            self.vphi = np.nan
            self.vcosth = np.nan
            self.Rmerge = np.nan
            self.Rmerge_proj = np.nan
            self.Vfinal = np.nan
        if self.sys_flag == 'radial_simple' or self.sys_flag == 'tangential' or self.sys_flag == 'radial_simple2' or self.sys_flag == 'tangential2':
            self.vphi = np.nan
            self.vcosth = np.nan
            self.Rmerge_proj = np.nan
            self.Vfinal = np.nan
            self.Apost = np.nan
            self.epost = np.nan
            self.R_proj=np.nan
            self.omega=np.nan

        data = [self.M2*u.kg.to(u.M_sun), self.Mns*u.kg.to(u.M_sun), self.Mhe*u.kg.to(u.M_sun), \
                self.Apre*u.m.to(u.R_sun), self.Apost*u.m.to(u.R_sun), self.epre, self.epost, self.d*u.m.to(u.Mpc), \
                self.R*u.m.to(u.kpc), self.R_proj*u.m.to(u.kpc), self.galcosth, self.galphi, \
                self.Vkick*u.m.to(u.km), self.phi, self.costh, self.omega, self.vphi, self.vcosth, \
                self.Tmerge*u.s.to(u.Gyr), self.Rmerge*u.m.to(u.kpc), self.Rmerge_proj*u.m.to(u.kpc), self.Vfinal*u.m.to(u.km), self.flag]
        return data


    def save_evolution(self, filename):
        '''
        If called, will save the evolution of a given system for plotting orbital trajectory through galaxy
        Format: [t, X, Y, Z, Vx, Vy, Vz]
        The initial values are saved as the first item of the file
        '''
        # save initial conditions
        initial = np.atleast_2d([self.Vkick,self.Mhe,self.Apre,self.Apost,self.Rmerge,self.Vxcirc0,self.Vycirc0,self.Vzcirc0])
        dfi = pd.DataFrame(initial, columns=['Vkick','Mhe','Apre','Apost','Rmerge','vx','vy','vz'])
        dfi.to_csv('evolution/'+filename+'_ini.dat', index=False)        

        # save evolution
        evolution = np.vstack([[self.t],[self.X],[self.Y],[self.Z],[self.Vx],[self.Vy],[self.Vz]]).T
        df = pd.DataFrame(evolution, columns=['t','x','y','z','vx','vy','vz'])
        df.to_csv('evolution/'+filename+'.dat', index=False)


