import scipy.optimize as op
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp


def smooth_f(x,a=1):
	return np.abs((x+a*np.logaddexp(x/a,-x/a))/2.0)


def T_prof_lin(h,dT,A,a,b):
	return 14.8-dT*(h-49)+A*np.exp(-smooth_f(h/a)**b)

def T_prof_lin(h,A,a,b):
	return A*(1-1/(1+np.exp((h-b)/a)))+13.1-0.0098*(h-124)

def T_prof_lin(h,T0,A,a):
	return T0-0.0098*h+A/(1+np.exp(h/a))

def T_prof_lin_2(h,T0,A,a):
	return T0-0.0098*h+A/(1+np.exp(h/a)**0.5)

def chi(p,f,x,y,dy):
	return (((f(x,*p)-y)/dy)**2).sum()


# h_p = np.array([1,49,106,124])
# T_p = np.array([10,16.3,15.8,15.2])
# dT_p = np.array([1,1,1,1])

# h_p = np.array([1,49,106,124])
# T_p = np.array([10,16.3,15.8,15.2])
# dT_p = np.array([1,1,1,1])

# h_p = np.array([4,106,124])
# T_p = np.array([10,15.8,15.2])
# dT_p = np.array([1,1,1])

# h_p = np.array([1,49,106,124])
# T_p = np.array([12,16.3,15.8,15.2])
# dT_p = np.array([0.25,0.25,0.25,0.25])

# h_p = np.array([0.2,1,106,124])
# T_p = np.array([8.7,10.2,13.5,13.1])
# dT_p = np.array([1,1,1,1])

h_p = np.array([0.2,1,49,106,124])
T_p = np.array([8.7,10.2,14.8,13.5,13.1])
dT_p = np.array([1,1,1,1,1])

hh = np.linspace(0,150,10001)
bounds = [(0,-50,0),(30,50,50)]




x = 16.62699400130137,-11.463539854542974, 1.218272068898033
print T_prof_lin(124,*x),(T_prof_lin(0.2,*x)-T_prof_lin(1,*x))/0.8

plt.plot(T_prof_lin_2(hh,*x),hh,marker="",linestyle=":")
plt.plot(T_p,h_p,marker=".",linestyle="")

x,dx= op.curve_fit(T_prof_lin,h_p,T_p,bounds=bounds,maxfev=10000)
print x
plt.plot(T_prof_lin_2(hh,*x),hh,marker="",linestyle=":")
plt.show()
