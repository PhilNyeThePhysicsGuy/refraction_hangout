from refraction import *
from raytrace import render_image_fast,ray_crossing
import georasters as gr
import numpy as np
import scipy.interpolate as interp
from pyproj import Geod,pj_ellps
from PIL import Image




class land_model(object):
    """
    This class is used to grab slices of the tarrain along great circle
    slices of the earth. 
    """
    def __init__(self,raster,ellps='WGS84',tide=0):
        NDV, xsize, ysize, GeoT, Projection, DataType = gr.get_geo_info(raster)

        # self._v = GeoT[0]+np.arange(xsize)*GeoT[1]
        # self._u = GeoT[3]+np.arange(ysize)*GeoT[5]
        self._v = -4.8791666667 + np.arange(xsize)*GeoT[1]
        self._u = 54.4350000000 + np.arange(ysize)*GeoT[5]

        self._geod = Geod(ellps=ellps)

        data = gr.load_tiff(raster)
        self._data = np.asarray(data,dtype=np.float32)
        self._data[self._data==np.iinfo(np.int16).min] = 0

        arg = np.argsort(self._u)
        self._data = self._data[arg,:].copy()
        self._u = self._u[arg].copy()

        self._terrain = interp.RegularGridInterpolator((self._u,self._v),self._data,bounds_error=False,fill_value=0.0)
        self._R0 = 6370997.0
        self._tide = tide

    @property
    def lats(self):
        return self._u

    @property
    def longs(self):
        return self._v

    @property
    def R0(self):
        return self._R0

    @property
    def data(self):
        return self._data

    def get_terrain(self,theta0,phi0,heading,dist):
        phi,theta,az = self._geod.fwd(phi0,theta0,heading,dist)
        height = self._terrain(np.vstack((theta,phi)).T)
        return height+self._tide

def render_land(png_data,d,rs,d_min,d_max,theta_i,phi_i,headings,angles,terrain,eye_level):
	n_v = len(angles)
	
	a_v = angles.max()-angles.min()
	i_min_angle = angles.argmin()

	hmax = terrain.data.max()
	print hmax,d_min

	exit()

	water = np.zeros(n_v,dtype=np.bool)
	land = np.zeros(n_v,dtype=np.bool)
	sky = np.zeros(n_v,dtype=np.bool)
	lh = np.zeros(n_v,dtype=np.bool)
	inds = np.zeros(n_v,dtype=np.int32)

	azs = np.zeros_like(d)
	theta_is = np.full_like(d,theta_i)
	phi_is = np.full_like(d,phi_i)

	if eye_level:
		i_horz = np.abs(angles)<(a_v/400.0)
	else:
		i_horz = np.array([])

	for i,heading in enumerate(headings):
		azs[:] = heading
		heights = terrain.get_terrain(theta_is,phi_is,azs,d)

		ray_crossing(rs,heights,water,land,sky,inds)
		png_data[i,water,2] = 160
		png_data[i,sky,1] = 180
		png_data[i,sky,2] = 255

		if np.any(land):
			land_inds = inds[land]

			ng = 100+(255-100)*(d[land_inds]/d_min)**(-4)
			nr = ng*(1-heights[land_inds]/hmax)
			png_data[i,land,0] = nr
			png_data[i,land,1] = ng


		if np.any(i_horz):
			png_data[i,i_horz,0] = 255
			png_data[i,i_horz,1] = 100
			png_data[i,i_horz,2] = 0			

		print heading,headings[-1],land.sum()

def sphere_image_fast(calc_args,image_name,ellps,theta_i,phi_i,h0,heading_mins,heading_maxs,n_v=200,f_length=2000,eye_level=False,alpha_horizon=0.0):

	a_h,a_v = get_angle_of_view(f_length)

	calc = CurveCalc(**calc_args)

	terrain = land_model("height_data/isle_of_man.tif",ellps)

	lats = np.kron(terrain.lats,np.ones_like(terrain.longs))
	longs = np.kron(np.ones_like(terrain.lats),terrain.longs)

	h,dists = gc_dist(theta_i,phi_i,lats,longs)

	d_min = dists.min()
	d_max = dists.max()

	d = np.arange(0,d_max,30)
	sigmas = d / calc.R0

	a_min = alpha_horizon - a_v/2.0
	a_max = alpha_horizon + a_v/2.0

	angles,dheading = np.linspace(a_min,a_max,n_v,retstep=True)
	sols = calc.solve_ivp(d_max,h0,alpha=angles,atol=1.1e-3,rtol=1.1e-5)

	rs = sols.sol(np.pi/2+sigmas)[:n_v,:].copy()
	rs -= calc.R0

	heading_mins = np.array(heading_mins)
	heading_maxs = np.array(heading_maxs)

	if heading_mins.ndim==0:
		heading_mins = heading_mins.reshape((1,))

	if heading_maxs.ndim==0:
		heading_maxs = heading_maxs.reshape((1,))

	if np.any(heading_maxs<heading_mins):
		raise ValueError

	for heading_min,heading_max in zip(heading_mins,heading_maxs):
		headings = np.arange(heading_min,heading_max,dheading)

		png_data = np.empty((len(headings),len(angles),3),dtype=np.uint8)
		png_data[...] = 0

		render_land(png_data,d,rs,d_min,d_max,theta_i,phi_i,headings,angles,terrain,eye_level)

		png_data = png_data.transpose((1,0,2))
		png_data = png_data[::-1,:,:]
		im = Image.fromarray(png_data,mode="RGB")
		im.save(image_name.format(a_v,heading_min,heading_max))

def flat_image_fast(calc_args,image_name,ellps,theta_i,phi_i,h0,heading_mins,heading_maxs,n_v=200,f_length=2000,eye_level=False,alpha_horizon=0):

	a_h,a_v = get_angle_of_view(f_length)

	calc = FlatCalc(**calc_args)
	terrain = land_model("height_data/isle_of_man.tif",ellps)

	lats = np.kron(terrain.lats,np.ones_like(terrain.longs))
	longs = np.kron(np.ones_like(terrain.lats),terrain.longs)

	h,dists = gc_dist(theta_i,phi_i,lats,longs)

	d_min = dists.min()
	d_max = dists.max()

	d = np.arange(0,d_max,30)
	sigmas = d/terrain.R0

	a_min = alpha_horizon - a_v/2.0
	a_max = alpha_horizon + a_v/2.0

	angles,dheading = np.linspace(a_min,a_max,n_v,retstep=True)
	sols = calc.solve_ivp(d_max,h0,alpha=angles,atol=1.1e-3,rtol=1.1e-5)

	rs = sols.sol(d)[:n_v,:].copy()

	heading_mins = np.array(heading_mins)
	heading_maxs = np.array(heading_maxs)

	if heading_mins.ndim==0:
		heading_mins = heading_mins.reshape((1,))

	if heading_maxs.ndim==0:
		heading_maxs = heading_maxs.reshape((1,))

	if np.any(heading_maxs<heading_mins):
		raise ValueError

	for heading_min,heading_max in zip(heading_mins,heading_maxs):
		headings = np.arange(heading_min,heading_max,dheading)

		png_data = np.empty((len(headings),len(angles),3),dtype=np.uint8)
		png_data[...] = 0

		render_land(png_data,d,rs,d_min,d_max,theta_i,phi_i,headings,angles,terrain,eye_level)

		png_data = png_data.transpose((1,0,2))
		png_data = png_data[::-1,:,:]
		im = Image.fromarray(png_data,mode="RGB")
		im.save(image_name.format(a_v,heading_min,heading_max))

def sphere_ray_diagram_fast(calc_args,image_name,ellps,theta_i,phi_i,h0,headings,n_v=200,f_length=2000,eye_level=False,alpha_horizon=0,lw=0.2,scale_y=True):

	a_h,a_v = get_angle_of_view(f_length)

	calc = CurveCalc(**calc_args)

	terrain = land_model("height_data/isle_of_man.tif",ellps)

	lats = np.kron(terrain.lats,np.ones_like(terrain.longs))
	longs = np.kron(np.ones_like(terrain.lats),terrain.longs)

	h,dists = gc_dist(theta_i,phi_i,lats,longs)

	d_min = dists.min()
	d_max = dists.max()

	d = int(np.ceil(d_max/1000.0))*1000.0
	d = np.arange(0,d,10)
	sigmas = d / terrain.R0

	a_min = alpha_horizon - a_v/2.0
	a_max = alpha_horizon + a_v/2.0

	angles = np.linspace(a_min,a_max,n_v)
	sols = calc.solve_ivp(d_max,h0,alpha=angles,atol=1.1e-7,rtol=1.1e-7)

	azs = np.zeros_like(d)
	theta_is = np.full_like(d,theta_i)
	phi_is = np.full_like(d,phi_i)

	water = np.zeros(n_v,dtype=np.bool)
	land = np.zeros(n_v,dtype=np.bool)
	sky = np.zeros(n_v,dtype=np.bool)
	inds = np.zeros(n_v,dtype=np.int32)

	rs = sols.sol(np.pi/2+sigmas)[:n_v,:].copy()
	c = np.cos(np.pi/2+sigmas-d_max/(2*terrain.R0))
	s = np.sin(np.pi/2+sigmas-d_max/(2*terrain.R0))

	y_shift = (terrain.R0*s).min()
	x_shift = max((rs*c).max(),(terrain.R0*c).max())
	x_min = min((rs*c).min(),(terrain.R0*c).min())
	x_width = x_shift-x_min

	for heading in headings:
		azs[:] = heading

		heights = terrain.get_terrain(theta_is,phi_is,azs,d)
		rs -= terrain.R0
		ray_crossing(rs,heights,water,land,sky,inds)
		rs += terrain.R0
		heights += terrain.R0

		i_horz = np.abs(angles).argmin()
		for i in range(n_v):
			if i == i_horz and eye_level:
				i_max = inds[i]
				plt.plot(rs[i,:i_max]*c[:i_max]-x_shift,
					     rs[i,:i_max]*s[:i_max]-y_shift,color="orange",linewidth=lw)
				continue

			if water[i]:
				i_max = inds[i]
				plt.plot(rs[i,:i_max]*c[:i_max]-x_shift,
					     rs[i,:i_max]*s[:i_max]-y_shift,color="blue",linewidth=lw)
				continue

			if land[i]:
				i_max = inds[i]
				plt.plot(rs[i,:i_max]*c[:i_max]-x_shift,
					     rs[i,:i_max]*s[:i_max]-y_shift,color="green",linewidth=lw)
				continue

			if sky[i]:
				plt.plot(rs[i,:]*c-x_shift,rs[i,:]*s-y_shift,color="cyan",linewidth=lw)
				continue

		plt.plot(c*heights-x_shift,s*heights-y_shift,color="green",linewidth=lw)
		plt.plot(c*terrain.R0-x_shift,s*terrain.R0-y_shift,color="blue",linewidth=lw)
		ax = plt.gca()
		for axis in ['top','bottom','left','right']:
			ax.spines[axis].set_linewidth(0.2)
		ax.tick_params(axis="both",direction="in",which="major",labelsize=5,width=0.2,length=1)
		ax.tick_params(axis="both",direction="in",which="minor",width=0.2,length=0.5)
		ax.tick_params(axis="x", pad=-6)
		ax.tick_params(axis="y", pad=-14)
		
		
		if scale_y:
			ax.set_ylim((0,1500))
		ax.set_xlim((-x_width,0))
		xticks = [-80000,-60000,-40000,-20000]
		yticks = range(100,1500,100)

		xlabels = [str(-x) for x in xticks]
		ylabels = [str(y) for y in yticks]
		plt.xticks(xticks,xlabels)
		plt.yticks(yticks,ylabels)
		ax.set_xticks(-np.arange(0,x_width,1000.0),minor=True)
		plt.savefig(image_name.format(heading))
		plt.clf()

def flat_ray_diagram_fast(calc_args,image_name,ellps,theta_i,phi_i,h0,headings,n_v=200,f_length=2000,eye_level=False,alpha_horizon=0,lw=0.2,save_opt={},scale_y=True):
	
	a_h,a_v = get_angle_of_view(f_length)

	calc = FlatCalc(**calc_args)

	terrain = land_model("height_data/isle_of_man.tif",ellps)

	lats = np.kron(terrain.lats,np.ones_like(terrain.longs))
	longs = np.kron(np.ones_like(terrain.lats),terrain.longs)

	h,dists = gc_dist(theta_i,phi_i,lats,longs)

	d_min = dists.min()
	d_max = dists.max()

	d = int(np.ceil(d_max/1000.0))*1000.0
	d = np.arange(0,d,10)

	a_min = alpha_horizon - a_v/2.0
	a_max = alpha_horizon + a_v/2.0

	angles = np.linspace(a_min,a_max,n_v)
	sols = calc.solve_ivp(d_max,h0,alpha=angles,atol=1.1e-3,rtol=1.1e-5)

	water = np.zeros(n_v,dtype=np.bool)
	land = np.zeros(n_v,dtype=np.bool)
	sky = np.zeros(n_v,dtype=np.bool)
	inds = np.zeros(n_v,dtype=np.int32)

	azs = np.zeros_like(d)
	theta_is = np.full_like(d,theta_i)
	phi_is = np.full_like(d,phi_i)

	y_shift = 50
	rs = sols.sol(d)[:n_v,:].copy()
	x_width = d_max

	for heading in headings:
		azs[:] = heading

		heights = terrain.get_terrain(theta_is,phi_is,azs,d)
		ray_crossing(rs,heights,water,land,sky,inds)

		i_horz = np.abs(angles).argmin()

		for i in range(n_v):
			if i == i_horz and eye_level:
				i_max = inds[i]
				plt.plot(-d[:i_max],rs[i,:i_max]+y_shift,color="orange",linewidth=lw)
				continue

			if water[i]:
				i_max = inds[i]
				plt.plot(-d[:i_max],rs[i,:i_max]+y_shift,color="blue",linewidth=lw)
				continue

			if land[i]:
				i_max = inds[i]
				plt.plot(-d[:i_max],rs[i,:i_max]+y_shift,color="green",linewidth=lw)
				continue

			if sky[i]:
				plt.plot(-d,rs[i,:]+y_shift,color="cyan",linewidth=lw)
				continue

		plt.plot(-d,heights+y_shift,color="green",linewidth=lw)
		plt.plot(-d,y_shift*np.ones_like(d),color="blue",linewidth=lw)
		ax = plt.gca()
		for axis in ['top','bottom','left','right']:
			ax.spines[axis].set_linewidth(0.2)
		ax.tick_params(axis="both",direction="in",which="major",labelsize=5,width=0.2,length=1)
		ax.tick_params(axis="both",direction="in",which="minor",width=0.2,length=0.5)
		ax.tick_params(axis="x", pad=-6)
		ax.tick_params(axis="y", pad=-12)
		
		
		if scale_y:
			ax.set_ylim((0,1000))
		ax.set_xlim((-x_width,0))
		xticks = [-80000,-60000,-40000,-20000]
		yticks = np.arange(100,1000,100)+y_shift

		xlabels = [str(-x) for x in xticks]
		ylabels = [str(y-y_shift) for y in yticks]
		plt.xticks(xticks,xlabels)
		plt.yticks(yticks,ylabels)
		ax.set_xticks(-np.arange(0,x_width,1000.0),minor=True)
		plt.savefig(image_name.format(heading),**save_opt)
		plt.clf()


res=2000

# location 1 27/10/17
theta_i, phi_i = 54.487375, -3.599760


h0 = 35 

# Temperature profiles
def T_prof(h):
	e1 = np.exp(h/1.5)
	e2 = np.exp(h/0.1)
	return (2/(1+e1))*0.1+(2/(1+e2))*0.15

calc_args = dict(T0=8.3,P0=103000,T_prof=T_prof)
calc = CurveCalc(**calc_args)

h = np.linspace(0,1000,10001)

plt.plot(calc.T(h)-273,h,color="blue")
plt.xlabel("Temperature (C)")
plt.ylabel("Height (m)")
plt.savefig("images/location_1/weather_profiles/Temperature_inf.svg")
plt.clf()

plt.plot(calc.P(h),h,color="green")
plt.xlabel("Pressure (Pa)")
plt.ylabel("Height (m)")
plt.savefig("images/location_1/weather_profiles/pressure_inf.svg")
plt.clf()

plt.plot(calc.n(h),h,color="red")
plt.xlabel("index of refraction")
plt.ylabel("Height (m)")
plt.savefig("images/location_1/weather_profiles/ior_inf.svg")
plt.clf()


plt.plot(calc.rho(h),h,color="orange")
plt.xlabel("density (kg/$\mathrm{m}^3$)")
plt.ylabel("Height (m)")
plt.savefig("images/location_1/weather_profiles/density_inf.svg")
plt.clf()


h = np.linspace(0,30,1001)

plt.plot(calc.T(h)-273,h,color="blue")
plt.xlabel("Temperature (C)")
plt.ylabel("Height (m)")
plt.savefig("images/location_1/weather_profiles/Temperature_inf_zoom.svg")
plt.clf()

plt.plot(calc.P(h),h,color="green")
plt.xlabel("Pressure (Pa)")
plt.ylabel("Height (m)")
plt.savefig("images/location_1/weather_profiles/pressure_inf_zoom.svg")
plt.clf()

plt.plot(calc.n(h),h,color="red")
plt.xlabel("index of refraction")
plt.ylabel("Height (m)")
plt.savefig("images/location_1/weather_profiles/ior_inf_zoom.svg")
plt.clf()


plt.plot(calc.rho(h),h,color="orange")
plt.xlabel("density (kg/$\mathrm{m}^3$)")
plt.ylabel("Height (m)")
plt.savefig("images/location_1/weather_profiles/density_inf_zoom.svg")
plt.clf()


calc_args = dict(T0=8.3,P0=103000)
calc = CurveCalc(**calc_args)

h = np.linspace(0,1000,10001)

plt.plot(calc.T(h)-273,h,color="blue")
plt.xlabel("Temperature (C)")
plt.ylabel("Height (m)")
plt.savefig("images/location_1/weather_profiles/Temperature.svg")
plt.clf()

plt.plot(calc.P(h),h,color="green")
plt.xlabel("Pressure (Pa)")
plt.ylabel("Height (m)")
plt.savefig("images/location_1/weather_profiles/pressure.svg")
plt.clf()

plt.plot(calc.n(h),h,color="red")
plt.xlabel("index of refraction")
plt.ylabel("Height (m)")
plt.savefig("images/location_1/weather_profiles/ior.svg")
plt.clf()


plt.plot(calc.rho(h),h,color="orange")
plt.xlabel("density (kg/$\mathrm{m}^3$)")
plt.ylabel("Height (m)")
plt.savefig("images/location_1/weather_profiles/density.svg")
plt.clf()


h = np.linspace(0,30,1001)

plt.plot(calc.T(h)-273,h,color="blue")
plt.xlabel("Temperature (C)")
plt.ylabel("Height (m)")
plt.savefig("images/location_1/weather_profiles/Temperature_zoom.svg")
plt.clf()

plt.plot(calc.P(h),h,color="green")
plt.xlabel("Pressure (Pa)")
plt.ylabel("Height (m)")
plt.savefig("images/location_1/weather_profiles/pressure_zoom.svg")
plt.clf()

plt.plot(calc.n(h),h,color="red")
plt.xlabel("index of refraction")
plt.ylabel("Height (m)")
plt.savefig("images/location_1/weather_profiles/ior_zoom.svg")
plt.clf()


plt.plot(calc.rho(h),h,color="orange")
plt.xlabel("density (kg/$\mathrm{m}^3$)")
plt.ylabel("Height (m)")
plt.savefig("images/location_1/weather_profiles/density_zoom.svg")
plt.clf()


calc_args = dict(T0=8.3,P0=103000,T_prof=T_prof)
calc = CurveCalc(**calc_args)

headings_mins = np.arange(235,259,3)
heading_maxs = headings_mins + 3

calc_args = dict(T0=8.3,P0=103000)
# sphere_image_fast(calc_args,"images/location_1/a/IOM_parts/IOM_location_1_a_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"sphere",theta_i,phi_i,h0,headings_mins,heading_maxs,res,f_length=2000,alpha_horizon=0.1,eye_level=True)
# sphere_ray_diagram_fast(calc_args,"images/location_1/a/ray_diagram_location_1_a_{}.svg","sphere",theta_i,phi_i,h0,[246,251,257],100,f_length=2000,alpha_horizon=0.1)

# flat_image_fast(calc_args,"images/location_1/c/IOM_parts/IOM_location_1_c_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"sphere",theta_i,phi_i,h0,headings_mins,heading_maxs,res,f_length=2000,alpha_horizon=0.3,eye_level=True)
# flat_ray_diagram_fast(calc_args,"images/location_1/c/ray_diagram_location_1_c_{}.svg","sphere",theta_i,phi_i,h0,[246,251,257],100,f_length=2000,alpha_horizon=0.3)

calc_args = dict(T0=8.3,P0=103000,T_prof=T_prof)
# sphere_image_fast(calc_args,"images/location_1/b/IOM_parts/IOM_location_1_b_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"sphere",theta_i,phi_i,h0,headings_mins,heading_maxs,res,f_length=2000,alpha_horizon=0.1,eye_level=True)
# sphere_ray_diagram_fast(calc_args,"images/location_1/b/ray_diagram_location_1_b_{}.svg","sphere",theta_i,phi_i,h0,[246,251,257],100,f_length=2000,alpha_horizon=0.1)
# sphere_ray_diagram_fast(calc_args,"images/location_1/b/ray_diagram_dense_location_1_b_{}.svg","sphere",theta_i,phi_i,h0,[246,251,257],2000,f_length=2000,alpha_horizon=0.1,lw=0.05)


# flat_image_fast(calc_args,"images/location_1/d/IOM_parts/IOM_location_1_d_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"sphere",theta_i,phi_i,h0,headings_mins,heading_maxs,res,f_length=2000,alpha_horizon=0.3,eye_level=True)
# flat_ray_diagram_fast(calc_args,"images/location_1/d/ray_diagram_location_1_d_{}.svg","sphere",theta_i,phi_i,h0,[246,251,257],100,f_length=2000,alpha_horizon=0.3)

calc_args = dict(n_funcs=(lambda t,r:1.0,lambda t,r:0.0,lambda t,r:0.0))
# sphere_image_fast(calc_args,"images/location_1/e/IOM_parts/IOM_location_1_e_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"sphere",theta_i,phi_i,h0,headings_mins,heading_maxs,res,f_length=2000,alpha_horizon=0.1,eye_level=True)
# sphere_ray_diagram_fast(calc_args,"images/location_1/e/ray_diagram_location_1_e_{}.svg","sphere",theta_i,phi_i,h0,[246,251,257],100,f_length=2000,alpha_horizon=0.1)

# flat_image_fast(calc_args,"images/location_1/f/IOM_parts/IOM_location_1_f_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"sphere",theta_i,phi_i,h0,headings_mins,heading_maxs,res,f_length=2000,alpha_horizon=0.25,eye_level=True)
# flat_ray_diagram_fast(calc_args,"images/location_1/f/ray_diagram_location_1_f_{}.svg","sphere",theta_i,phi_i,h0,[246,251,257],100,f_length=2000,alpha_horizon=0.2)

# R0 = (7.0/6.0) * 6370997.0
# calc_args = dict(n_funcs=(lambda t,r:1.0,lambda t,r:0.0,lambda t,r:0.0),R0=R0)
# sphere_image_fast(calc_args,"images/location_1/g/IOM_parts/IOM_location_1_g_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"sphere",theta_i,phi_i,h0,headings_mins,heading_maxs,res,f_length=2000,alpha_horizon=0.1,eye_level=True)
# sphere_ray_diagram_fast(calc_args,"images/location_1/g/ray_diagram_location_1_g_{}.svg","sphere",theta_i,phi_i,h0,[246,251,257],100,f_length=2000,alpha_horizon=0.1)



# Location 2 21/04/18 
# theta_i, phi_i = 54.490729, -3.608613
theta_i, phi_i = 54.489053, -3.604198

h0 = 3



def smooth_f(x,a=1):
	return np.abs((x+a*np.logaddexp(x/a,-x/a))/2.0)
# Temperature model
def T_prof(h):
	return A*np.exp(-smooth_f(h/a)**b)


h_p_bouy = np.array([1])
T_p_bouy = np.array([10])

h_p = np.array([49,106,124])
T_p = np.array([16.3,15.8,15.2])

dT,A,a,b = 0.0098,-11.3, 4.0,0.5 # 8
calc_args = dict(T0=14.5,P0=100600,h0=124,dT=dT,T_prof=T_prof)

calc = CurveCalc(**calc_args)
h = np.linspace(0,150,1001)
plt.plot(calc.T(h)-273,h,color="blue")
plt.plot(T_p_bouy,h_p_bouy,marker=".",linestyle="",color="green")
plt.plot(T_p-1,h_p,marker=".",linestyle="",color="green")
plt.plot(T_p,h_p,marker=".",linestyle="",color="red")
plt.xlabel("Temperature (C)")
plt.ylabel("Height (m)")
plt.savefig("images/location_2/weather_profiles/Temperature_model+data.svg")
plt.clf()

plt.plot(calc.n(h),h,color="red")
plt.xlabel("index of refraction")
plt.ylabel("Height (m)")
plt.savefig("images/location_2/weather_profiles/ior_calculation.svg")
plt.clf()


headings_mins = np.arange(235,259,3)
heading_maxs = headings_mins + 3

calc_args = dict(n_funcs=(lambda t,r:1.0,lambda t,r:0.0,lambda t,r:0.0))
# sphere_image_fast(calc_args,"images/location_2/e/IOM_parts/IOM_location_2_e_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"sphere",theta_i,phi_i,h0,headings_mins,heading_maxs,res,f_length=2000,alpha_horizon=0.1,eye_level=True)
# sphere_ray_diagram_fast(calc_args,"images/location_2/e/ray_diagram_location_2_e_{}.svg","sphere",theta_i,phi_i,h0,[246,251,257],100,f_length=2000,alpha_horizon=0.1)

# flat_image_fast(calc_args,"images/location_2/f/IOM_parts/IOM_location_2_f_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"sphere",theta_i,phi_i,h0,headings_mins,heading_maxs,res,f_length=2000,alpha_horizon=0.3,eye_level=True)
# flat_ray_diagram_fast(calc_args,"images/location_2/f/ray_diagram_location_2_f_{}.svg","sphere",theta_i,phi_i,h0,[246,251,257],100,f_length=2000,alpha_horizon=0.3)


# calc_args = dict(T0=14.5,P0=100600,h0=124,dT=dT)
# sphere_image_fast(calc_args,"images/location_2/a/IOM_parts/IOM_location_2_a_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"sphere",theta_i,phi_i,h0,headings_mins,heading_maxs,res,f_length=2000,alpha_horizon=0.15,eye_level=True)
# sphere_ray_diagram_fast(calc_args,"images/location_2/a/ray_diagram_location_2_a_{}.svg","sphere",theta_i,phi_i,h0,[246,251,257],100,f_length=2000,alpha_horizon=0.15)

# flat_image_fast(calc_args,"images/location_2/c/IOM_parts/IOM_location_2_c_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"sphere",theta_i,phi_i,h0,headings_mins,heading_maxs,res,f_length=2000,alpha_horizon=0.3,eye_level=True)
# flat_ray_diagram_fast(calc_args,"images/location_2/c/ray_diagram_location_2_c_{}.svg","sphere",theta_i,phi_i,h0,[246,251,257],100,f_length=2000,alpha_horizon=0.3)

# calc_args = dict(T0=14.5,P0=100600,h0=124,dT=dT,T_prof=T_prof)

# flat_image_fast(calc_args,"images/location_2/d/IOM_parts/IOM_location_2_d_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"sphere",theta_i,phi_i,h0,headings_mins,heading_maxs,res,f_length=2000,alpha_horizon=0.32,eye_level=True)
# flat_ray_diagram_fast(calc_args,"images/location_2/d/ray_diagram_location_2_d_{}.svg","sphere",theta_i,phi_i,h0,[246,251,257],100,f_length=2000,alpha_horizon=0.3)

# headings_mins = np.arange(235,259,3)
# heading_maxs = headings_mins + 3
# sphere_image_fast(calc_args,"images/location_2/b/IOM_parts/IOM_location_2_b_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"sphere",theta_i,phi_i,h0,headings_mins,heading_maxs,2500,f_length=2000,alpha_horizon=0.2,eye_level=True)

# headings_mins = np.arange(235,259,1)
# heading_maxs = headings_mins + 1
# sphere_image_fast(calc_args,"images/location_2/b/IOM_parts/IOM_location_2_b_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"sphere",theta_i,phi_i,h0,headings_mins,heading_maxs,res,f_length=3000,alpha_horizon=0.2,eye_level=True)
# sphere_ray_diagram_fast(calc_args,"images/location_2/b/ray_diagram_location_2_b_{}.svg","sphere",theta_i,phi_i,h0,[246],100,f_length=2000,alpha_horizon=0.2)

# headings_mins = np.arange(251,259,0.25)
# heading_maxs = headings_mins + 0.25
# sphere_image_fast(calc_args,"images/location_2/b/IOM_parts_north_zoom/IOM_location_2_b_vfov_{{:.5f}}_part_{}_{{:.2f}}_{{:.2f}}.png".format(h0),"sphere",theta_i,phi_i,h0,headings_mins,heading_maxs,res,f_length=16000,alpha_horizon=0.15,eye_level=True)
# sphere_ray_diagram_fast(calc_args,"images/location_2/b/ray_diagram_location_2_b_{}.svg","sphere",theta_i,phi_i,h0,[251],100,f_length=16000,alpha_horizon=0.15)
# sphere_ray_diagram_fast(calc_args,"images/location_2/b/ray_diagram_location_2_b_{}.svg","sphere",theta_i,phi_i,h0,[100,257],100,f_length=64000,alpha_horizon=0.117)



h0=35
# headings_mins = np.arange(235,259,3)
# heading_maxs = headings_mins + 3
# sphere_image_fast(calc_args,"images/location_2/g/IOM_parts/IOM_location_2_g_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"sphere",theta_i,phi_i,h0,headings_mins,heading_maxs,res,f_length=2000,alpha_horizon=0.1,eye_level=True)
# sphere_ray_diagram_fast(calc_args,"images/location_2/g/ray_diagram_location_2_g_{}.svg","sphere",theta_i,phi_i,h0,[246],100,f_length=2000,alpha_horizon=0.1)
# sphere_ray_diagram_fast(calc_args,"images/location_2/g/ray_diagram_location_2_g_{}.svg","sphere",theta_i,phi_i,h0,[251],100,f_length=8000,alpha_horizon=-0.05)
# sphere_ray_diagram_fast(calc_args,"images/location_2/g/ray_diagram_location_2_g_{}.svg","sphere",theta_i,phi_i,h0,[100,257],100,f_length=16000,alpha_horizon=-0.08)
