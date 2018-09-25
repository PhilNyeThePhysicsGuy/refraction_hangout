import scipy.interpolate as interp
import scipy.optimize as op
from scipy.integrate import solve_bvp,solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from scipy.optimize import newton,bisect
import numpy as np
import matplotlib.pyplot as plt


def get_angle_of_view(f,h=36,v=24):
    return np.rad2deg(2*np.arctan2(h,2*f)),np.rad2deg(2*np.arctan2(v,2*f))

def get_angles_of_view(f,n_v,h=36,v=24):
    n_h = int(n_v*(float(h)/v))
    hh = np.linspace(-h//2,h//2,n_h)
    vv = np.linspace(-v//2,v//2,n_v)
    return np.rad2deg(np.arctan2(hh,f)),np.rad2deg(np.arctan2(vv,f))


def shift_coordinates(src_lats, src_longs, dest_lats, dest_longs):
    src_lats = np.deg2rad(src_lats)
    src_longs = np.deg2rad(src_longs)
    dest_lats =np.deg2rad(dest_lats)
    dest_longs = np.deg2rad(dest_longs)

    R = np.array([[np.cos(src_lats)*np.cos(src_longs),np.cos(src_lats)*np.sin(src_longs) ,-np.sin(src_lats)],
                 [-np.sin(src_longs)                 , np.cos(src_longs)                 ,                0],
                 [ np.cos(src_longs)*np.sin(src_lats), np.sin(src_lats)*np.sin(src_longs), np.cos(src_lats)]])

    v = np.array([np.sin(dest_lats)*np.cos(dest_longs),
                  np.sin(dest_lats)*np.sin(dest_longs),
                  np.cos(dest_lats)])
    v = R.dot(v)
    print v
    dest_lats = np.arctan2(np.sqrt(v[1]**2+v[0]**2),v[2])
    dest_longs = np.arctan2(v[1],v[0])
    return np.rad2deg(dest_lats),np.rad2deg(dest_longs)


def gc_dist(src_lats, src_longs, dest_lats, dest_longs,EARTH_RADIUS=6370997.0):
    """Calculate distance between (effectively) two Series of points."""
    # Convert from degrees to radians.
    src_lats = np.deg2rad(src_lats)
    src_longs = np.deg2rad(src_longs)
    dest_lats =np.deg2rad(dest_lats)
    dest_longs = np.deg2rad(dest_longs)

    # grab np.sines
    src_lat_sin = np.sin(src_lats)
    dest_lat_sin = np.sin(dest_lats)
    # grab np.conp.sines
    src_lat_cos =np.cos(src_lats)
    dest_lat_cos = np.cos(dest_lats)

    lng_deltas = dest_longs - src_longs
    lng_delta_cos = np.cos(lng_deltas)
    lng_delta_sin = np.sin(lng_deltas)

    d = EARTH_RADIUS * np.arctan2(
    np.sqrt(
        (dest_lat_cos * lng_delta_sin) ** 2 +
        (src_lat_cos * dest_lat_sin - src_lat_sin * dest_lat_cos * lng_delta_cos) ** 2
    ),
    src_lat_sin * dest_lat_sin + src_lat_cos * dest_lat_cos * lng_delta_cos
    )

    y = lng_delta_sin * dest_lat_cos
    x = src_lat_cos * dest_lat_sin - src_lat_sin * dest_lat_cos * lng_delta_cos

    theta = np.arctan2(y,x)

    return (theta+2*np.pi)%(2*np.pi),d


def fl_dist(src_lats, src_longs, dest_lats, dest_longs,EARTH_RADIUS=6371000.0):
    """Calculate distance between (effectively) two Series of points."""
    # Convert from degrees to radians.
    src_lats = np.deg2rad(src_lats)
    src_longs = np.deg2rad(src_longs)
    dest_lats =np.deg2rad(dest_lats)
    dest_longs = np.deg2rad(dest_longs)

    # grab np.sines
    src_lat_sin = np.sin(src_lats)
    dest_lat_sin = np.sin(dest_lats)
    # grab np.conp.sines
    src_lat_cos =np.cos(src_lats)
    dest_lat_cos = np.cos(dest_lats)

    lng_deltas = dest_longs - src_longs
    lng_delta_cos = np.cos(lng_deltas)
    lng_delta_sin = np.sin(lng_deltas)

    d = EARTH_RADIUS * np.arctan2(
    np.sqrt(
        (dest_lat_cos * lng_delta_sin) ** 2 +
        (src_lat_cos * dest_lat_sin - src_lat_sin * dest_lat_cos * lng_delta_cos) ** 2
    ),
    src_lat_sin * dest_lat_sin + src_lat_cos * dest_lat_cos * lng_delta_cos
    )

    theta = np.arctan2(lng_delta_sin * dest_lat_cos,src_lat_cos * dest_lat_sin
     - src_lat_sin * dest_lat_cos * lng_delta_cos)

    return theta,d


def mi_to_km(d):
    return 1.609344*d

def km_to_m(d):
    return 1000.0*d

def ft_to_m(d):
    return d*0.3048

def m_to_ft(d):
    return d/0.3048

def mi_to_m(d):
    return 1609.344*d

def m_to_mi(d):
    return d/1609.344

def polar_angle(r,theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R,dR = r(theta)
    dydx = (dR*s+R*c)/(dR*c-R*s)
    return np.arctan2(dydx,1)

def effective_curvature(phi):
    phi = np.rad2deg(phi)
    Ra = 6.37814*10**6
    Rb = 6.35675*10**6
    return (Ra**2)*Rb/((Ra*np.cos(phi))**2+(Rb*np.sin(phi))**2)


class EulerEquations(object):
    def __init__(self):
        pass

    def solve_bvp(self,a,b,ha,hb):
        def bc(ya,yb):
            return np.array([ya[0]-ha,yb[0]-hb])

        x = np.linspace(a,b,100)
        y = np.zeros((2,100))
        y[0,:] = ha+(hb-ha)*(x-a)/(b-a)
        y[1,:] = (hb-ha)/(b-a)
        self._yout = np.zeros_like(y)
        return solve_bvp(self,bc,x,y,max_nodes=100000,tol=1e-7)    

    def solve_ivp(self,a,b,h,dh,**kwargs):
        y0 = np.array([h,dh])
        self._yout = np.zeros_like(y0.reshape((2,-1)))
        return solve_ivp(self,(a,b),y0.ravel(),dense_output=True,**kwargs)

class EulerEquationsEuclid(EulerEquations):
    def __init__(self,n,dndx,dndy,args=()):
        self._n = n
        self._dndx = dndx
        self._dndy = dndy
        self._args = args

    def __call__(self,x,yin): # object(args,...)
        shape0 = yin.shape
        yin = yin.reshape((2,-1))
        try:
            self._yout[...] = yin[...]
        except ValueError:
            self._yout = yin.copy()
        y,dydx = yin[0],yin[1]

        n_val = self._n(x,y,*self._args)
        dndx_val = self._dndx(x,y,*self._args)
        dndy_val = self._dndy(x,y,*self._args)

        self._yout[0] = dydx
        self._yout[1] = (1+dydx**2)*(dndy_val-dydx*dndx_val)/n_val

        return self._yout.reshape(shape0)

class EulerEquationsPolar(EulerEquations):
    def __init__(self,n,dndtheta,dndr,args=()):
        self._n = n
        self._dndtheta = dndtheta
        self._dndr = dndr
        self._args = args    


    def __call__(self,theta,yin):
        if np.any(np.isnan(yin)):
            raise ValueError
        shape0 = yin.shape
        yin = yin.reshape((2,-1))
        try:
            self._yout[...] = yin[...]
        except ValueError:
            self._yout = yin.copy()
            
        r,dr = yin[0],yin[1]
        n_val = self._n(theta,r,*self._args)
        dndtheta_val = self._dndtheta(theta,r,*self._args)
        dndr_val = self._dndr(theta,r,*self._args)

        r2 = r**2
        dr2 = dr**2

        self._yout[0] = dr
        self._yout[1] = (n_val*r*(r2+2*dr)+(r2+dr2)*(r2*dndr_val-dr*dndtheta_val))/(n_val * r2)
        # yout[1] = ((-dndtheta_val)*dr*(dr2 + r2) + r*(r2*(n_val + dndr_val*r) + dr2*(2*n_val + dndr_val*r)))/(n_val*r2)

        return self._yout.reshape(shape0)

class CurveCalc(object):
    def __init__(self,T0=15.0,P0=101325.0,h0=0.0,g=9.8076,dT=None,moist_lapse_rate=False,
                ll=545,T_prof=None,phi=None,R0=6370997.0,T_prof_args=(),n_funcs=None):


        if phi is None:
            self._R0 = R0
        else:
            R0 = effective_curvature(phi)
            self._R0 = R0

        if n_funcs is None:
                    
            T0 = max(T0,0)
            e = 611.21*np.exp((18.678-T0/234.5)*(T0/(557.14+T0)))
            T0 += 273
            eps = 0.622
            cpd = 1003.5
            Hv = 2501000
            R = 287.058
            r = eps*e/(P0-e)

            if dT is None:
                if moist_lapse_rate:
                    dT = g*(R*T0**2+Hv*r*T0)/(cpd*R*T0**2+r*eps*Hv**2)
                else:
                    dT = 0.0098
                


            if T_prof is not None:
                T_prof0 = T_prof(h0+R0,*T_prof_args)
                T = lambda r:T0-dT*(r-(h0+R0))+(T_prof(r-R0,*T_prof_args)-T_prof0)
                dTdr = lambda r:-dT+(T_prof((r-R0)+1.1e-7,*T_prof_args)-T_prof((r-R0)-1.1e-7,*T_prof_args))/(2.2e-7)
            else:
                T = lambda r:T0-dT*(r-(h0+R0))
                dTdr = lambda r:-dT

            dPdh = lambda r,P:-g*P/(R*T(r))

            sol = solve_ivp(dPdh,(R0+h0,R0-10000),np.array([P0]),atol=1e-7,rtol=1e-13)

            P = sol.y[0,-1]
            sol = solve_ivp(dPdh,(R0-10000,R0+10000),np.array([P]),dense_output=True,atol=1e-7,rtol=1e-13)


            def drhodr(r):
                t = T(r)
                p = sol.sol(r)[0]
                dpdr = -g*p/(R*t)
                return (dpdr*t-dTdr(r)*p)/(R*t**2)

            rho = lambda r:sol.sol(r)[0]/(R*T(r))


            deltan = 0.0002879*(1+0.0000567/ll**2)
            n = lambda theta,r:(1+rho(r)*deltan)
            dndr = lambda theta,r:drhodr(r)*deltan
            dndtheta = lambda theta,r:0.0


            self._rho = rho
            self._dT = dT
            self._n = n
            self._P = lambda r:sol.sol(r)[0]
            self._T = T
        else:
            n,dndr,dndtheta = n_funcs

        self._ee = EulerEquationsPolar(n,dndtheta,dndr)

    @property
    def dT(self):
        return self._dT

    def T(self,h):
        return self._T(self._R0+h)

    def P(self,h):
        return self._P(self._R0+h)

    def n(self,h):
        return self._n(0.0,self._R0+h)

    def rho(self,h):
        return self._rho(self._R0+h)

    @property
    def R0(self):
        return self._R0

    def solve_bvp(self,d,ha,hb):
        theta_a = np.pi/2
        theta_b = np.pi/2+d/self._R0
        # print d
        R_a = (self.R0+ha)
        R_b = (self.R0+hb)
        sol = self._ee.solve_bvp(theta_a,theta_b,R_a,R_b)

        return sol

    def solve_ivp(self,d,h,dh=None,alpha=None,R0=None,**kwargs):
        theta_a = np.pi/2
        theta_b = np.pi/2+d/self._R0
        h = np.array(h)

        if R0 is None:
            R_a = (self.R0+h)
        else:
            R_a = R0+h
            
        if alpha is not None:
            dR_a = np.array(R_a*np.tan(np.deg2rad(alpha)))
        else:
            dR_a = np.array(dh)

        if R_a.ndim == 0 and dR_a.ndim > 0:
            R_a = np.ones_like(dR_a)*R_a
        elif R_a.ndim > 0 and dR_a.ndim == 0:
            dR_a = np.ones_like(R_a)*dR_a
        elif R_a.ndim > 0 and dR_a.ndim > 0:
            if len(R_a) != len(dR_a):
                raise ValueError("number of initial positions and slops must be equal.")

        sol = self._ee.solve_ivp(theta_a,theta_b,R_a,dR_a,**kwargs)

        return sol

    def solve_hidden(self,d,h):
        def der(d):
            
            sol = self.solve_bvp(d,h,0)
            return sol.yp[0,-1]

        dh = newton(der,1000)
        sol = self.solve_bvp(dh,h,0)

        Reff=newton(lambda R:np.cos(dh/R)-R/(R+h),self.R0,tol=1.1e-3)
        k = 1-self.R0/Reff
        return dh,k,self.solve_ivp(d,h,sol.yp[0,0])

class FlatCalc(object):
    def __init__(self,T0=15.0,P0=101325.0,h0=0.0,g=9.81,dT=None,moist_lapse_rate=False,
                ll=545,T_prof=None,T_prof_args=(),n_funcs=None):

        if n_funcs is None:
            T0 = max(T0,0)
            e = 611.21*np.exp((18.678-T0/234.5)*(T0/(557.14+T0)))
            T0 += 273
            eps = 0.622
            cpd = 1003.5
            Hv = 2501000
            R = 287.058
            r = eps*e/(P0-e)

            if dT is None:
                if moist_lapse_rate:
                    dT = g*(R*T0**2+Hv*r*T0)/(cpd*R*T0**2+r*eps*Hv**2)
                else:
                    dT = 0.0098

            if T_prof is not None:
                T = lambda r:T0-dT*(r-h0)+T_prof(r,*T_prof_args)
                dTdr = lambda r:-dT+(T_prof(r+1.1e-7,*T_prof_args)-T_prof(r-1.1e-7,*T_prof_args))/(2.2e-7)
            else:
                T = lambda r:T0-dT*(r-h0)
                dTdr = lambda r:-dT

            dPdh = lambda r,P:-g*P/(R*T(r))

            sol = solve_ivp(dPdh,(h0,-10000),np.array([P0]))

            P = sol.y[0,-1]
            sol = solve_ivp(dPdh,(-10000,10000),np.array([P]),dense_output=True)

            def drhody(r):
                t = T(r)
                p = sol.sol(r)[0]
                dpdr = -g*p/(R*t)
                return (dpdr*t-dTdr(r)*p)/(R*t**2)

            rho = lambda r:sol.sol(r)[0]/(R*T(r))


            deltan = 0.0002879*(1+0.0000567/ll**2)
            n = lambda theta,r:(1+rho(r)*deltan)
            dndr = lambda theta,r:drhodr(r)*deltan
            dndtheta = lambda theta,r:0.0


            deltan = 0.0002879*(1+0.0000567/ll**2)
            n = lambda theta,r:(1+rho(r)*deltan)
            dndy = lambda theta,r:drhody(r)*deltan
            dndx = lambda theta,r:0.0
            
            self._rho = rho
            self._dT = dT
            self._n = n
            self._P = lambda h:P(h)[0]
            self._T = T
        else:
            n,dndy,dndx = n_funcs




        self._ee = EulerEquationsEuclid(n,dndx,dndy)


    @property
    def R0(self):
        return 0

    @property
    def dT(self):
        return self._dT

    def T(self,h):
        return self._T(h)

    def P(self,h):
        return self._P(h)

    def n(self,theta,h):
        return self._n(theta,h)

    def solve_bvp(self,d,ha,hb):
        return self._ee.solve_bvp(0,d,ha,hb)

    def solve_ivp(self,d,h,dh=None,alpha=None,**kwargs):
        if alpha is not None:
            dh = np.tan(np.deg2rad(alpha))
        
        h = np.array(h)
        if h.ndim == 0 and dh.ndim > 0:
            h = np.ones_like(dh)*h
        elif h.ndim > 0 and dh.ndim == 0:
            dh = np.ones_like(h)*dh
        elif h.ndim > 0 and dh.ndim > 0:
            if len(h) != len(dh):
                raise ValueError("number of initial positions and slops must be equal.")


        return self._ee.solve_ivp(0,d,h,dh,**kwargs)


