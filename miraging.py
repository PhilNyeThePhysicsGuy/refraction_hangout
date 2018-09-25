# from refraction import *
from raytrace import render_image_fast,ray_crossing
from refraction_render.renderers import Scene,Renderer_35mm
from refraction_render.calcs import CurveCalc,FlatCalc

from pyproj import Geod
import numpy as np
import os


def sphere_image_fast(calc_args,image_name,theta_i,phi_i,h0,heading_mins,heading_maxs,n_v=200,f_length=2000,eye_level=False,alpha_horizon=0.0):

	R0 = 6371000.0
	a_h,a_v = get_angle_of_view(f_length)

	calc = CurveCalc(**calc_args)


	terrain = land_model("height_data/isle_of_man.tif")

	lats = np.kron(np.rad2deg(terrain.lats),np.ones_like(terrain.longs))
	longs = np.kron(np.ones_like(terrain.lats),np.rad2deg(terrain.longs))

	h,dists = gc_dist(theta_i,phi_i,lats,longs)

	d_min = dists.min()
	d_max = dists.max()

	d = np.arange(0,d_max,10)
	sigmas = d / R0

	a_min = alpha_horizon - a_v/2.0
	a_max = alpha_horizon + a_v/2.0

	angles,dheading = np.linspace(a_min,a_max,n_v,retstep=True)
	sols = calc.solve_ivp(d_max,h0,alpha=angles,atol=1.1e-3,rtol=1.1e-5)

	rs = sols.sol(np.pi/2+sigmas)[:n_v,:].copy()
	rs -= R0

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
		render_image_fast(d,sigmas,rs,d_min,d_max,theta_i,phi_i,headings,angles,
					terrain,image_name.format(a_v,heading_min,heading_max),eye_level)

def flat_image_fast(calc_args,image_name,theta_i,phi_i,h0,heading_mins,heading_maxs,n_v=200,f_length=2000,eye_level=False,alpha_horizon=0):

	R0 = 6371000.0
	a_h,a_v = get_angle_of_view(f_length)

	calc = FlatCalc(**calc_args)
	terrain = land_model("height_data/isle_of_man.tif")

	lats = np.kron(np.rad2deg(terrain.lats),np.ones_like(terrain.longs))
	longs = np.kron(np.ones_like(terrain.lats),np.rad2deg(terrain.longs))

	h,dists = gc_dist(theta_i,phi_i,lats,longs)

	d_min = dists.min()
	d_max = dists.max()

	d = np.arange(0,d_max,10)
	sigmas = d/R0

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
		render_image_fast(d,sigmas,rs,d_min,d_max,theta_i,phi_i,headings,angles,
					terrain,image_name.format(a_v,heading_min,heading_max),eye_level)


def sphere_ray_diagram_fast(calc_args,image_name,theta_i,phi_i,h0,headings,n_v=200,f_length=2000,eye_level=False,alpha_horizon=0,lw=0.2):

	R0 = 6371000.0
	a_h,a_v = get_angle_of_view(f_length)

	calc = CurveCalc(**calc_args)

	terrain = land_model("height_data/isle_of_man.tif")

	lats = np.kron(np.rad2deg(terrain.lats),np.ones_like(terrain.longs))
	longs = np.kron(np.ones_like(terrain.lats),np.rad2deg(terrain.longs))

	h,dists = gc_dist(theta_i,phi_i,lats,longs)

	d_min = dists.min()
	d_max = dists.max()

	d = int(np.ceil(d_max/1000.0))*1000.0
	d = np.arange(0,d,10)
	sigmas = d / R0

	a_min = alpha_horizon - a_v/2.0
	a_max = alpha_horizon + a_v/2.0

	angles = np.linspace(a_min,a_max,n_v)
	sols = calc.solve_ivp(d_max,h0,alpha=angles,atol=1.1e-3,rtol=1.1e-5)

	water = np.zeros(n_v,dtype=np.bool)
	land = np.zeros(n_v,dtype=np.bool)
	sky = np.zeros(n_v,dtype=np.bool)
	inds = np.zeros(n_v,dtype=np.int32)

	rs = sols.sol(np.pi/2+sigmas)[:n_v,:].copy()
	c = np.cos(np.pi/2+sigmas-d_max/(2*R0))
	s = np.sin(np.pi/2+sigmas-d_max/(2*R0))

	for heading in headings:

		sigmas,heights = terrain.get_terrain_along_heading(theta_i,phi_i,heading,sigmas=sigmas)
		rs -= R0
		heights -= R0
		ray_crossing(rs,heights,water,land,sky,inds)
		rs += R0
		heights += R0

		i_horz = np.abs(angles).argmin()
		y_min = (R0*s).min()
		x_max = (R0*c).max()
		for i in range(n_v):
			if i == i_horz and eye_level:
				i_max = inds[i]
				plt.plot(rs[i,:i_max]*c[:i_max]-x_max,
					     rs[i,:i_max]*s[:i_max]-y_min,color="orange",linewidth=lw)
				continue

			if water[i]:
				i_max = inds[i]
				plt.plot(rs[i,:i_max]*c[:i_max]-x_max,
					     rs[i,:i_max]*s[:i_max]-y_min,color="blue",linewidth=lw)
				continue

			if land[i]:
				i_max = inds[i]
				plt.plot(rs[i,:i_max]*c[:i_max]-x_max,
					     rs[i,:i_max]*s[:i_max]-y_min,color="green",linewidth=lw)
				continue

			if sky[i]:
				plt.plot(rs[i,:]*c-x_max,rs[i,:]*s-y_min,color="cyan",linewidth=lw)
				continue

		plt.plot(c*heights-x_max,s*heights-y_min,color="green",linewidth=lw)
		plt.plot(c*R0-x_max,s*R0-y_min,color="blue",linewidth=lw)
		ax = plt.gca()
		ylim = ax.get_ylim()
		xlim = ax.get_xlim()
		ax.set_xticks(np.arange(xlim[0],xlim[1],1000.0),minor=True)
		plt.grid(linestyle=":",linewidth=lw)
		plt.savefig(image_name.format(heading))
		plt.clf()

def flat_ray_diagram_fast(calc_args,image_name,theta_i,phi_i,h0,headings,n_v=200,f_length=2000,eye_level=False,alpha_horizon=0,save_opt={},lw=0.2):
	
	R0 = 6371000.0
	a_h,a_v = get_angle_of_view(f_length)

	calc = FlatCalc(**calc_args)

	terrain = land_model("height_data/isle_of_man.tif")

	lats = np.kron(np.rad2deg(terrain.lats),np.ones_like(terrain.longs))
	longs = np.kron(np.ones_like(terrain.lats),np.rad2deg(terrain.longs))

	h,dists = gc_dist(theta_i,phi_i,lats,longs)

	d_min = dists.min()
	d_max = dists.max()

	d = int(np.ceil(d_max/1000.0))*1000.0
	d = np.arange(0,d,10)
	sigmas = d / R0

	a_min = alpha_horizon - a_v/2.0
	a_max = alpha_horizon + a_v/2.0

	angles = np.linspace(a_min,a_max,n_v)
	sols = calc.solve_ivp(d_max,h0,alpha=angles,atol=1.1e-3,rtol=1.1e-5)

	water = np.zeros(n_v,dtype=np.bool)
	land = np.zeros(n_v,dtype=np.bool)
	sky = np.zeros(n_v,dtype=np.bool)
	inds = np.zeros(n_v,dtype=np.int32)

	rs = sols.sol(d)[:n_v,:].copy()

	for heading in headings:

		sigmas,heights = terrain.get_terrain_along_heading(theta_i,phi_i,heading,sigmas=sigmas)
		heights -= R0
		ray_crossing(rs,heights,water,land,sky,inds)

		i_horz = np.abs(angles).argmin()

		for i in range(n_v):
			if i == i_horz and eye_level:
				i_max = inds[i]
				plt.plot(d[:i_max],rs[i,:i_max],color="orange",linewidth=lw)
				continue

			if water[i]:
				i_max = inds[i]
				plt.plot(d[:i_max],rs[i,:i_max],color="blue",linewidth=lw)
				continue

			if land[i]:
				i_max = inds[i]
				plt.plot(d[:i_max],rs[i,:i_max],color="green",linewidth=lw)
				continue

			if sky[i]:
				plt.plot(d,rs[i,:],color="cyan",linewidth=lw)
				continue

		plt.plot(d,heights,color="green",linewidth=lw)
		plt.plot(d,np.zeros_like(d),color="blue",linewidth=lw)
		ax = plt.gca()
		ylim = ax.get_ylim()
		xlim = ax.get_xlim()
		plt.grid(linestyle=":",linewidth=lw)
		ax.set_xticks(np.arange(xlim[0],xlim[1],1000.0),minor=True)
		plt.savefig(image_name.format(heading),**save_opt)
		plt.clf()


def ships(calc,image_name):
	geod = Geod(ellps="sphere")
	s = Scene()

	d1 = 14000
	d2 = 13000

	theta_i, phi_i = 54.487375, -3.599760
	phi_1,theta_1,b_az = geod.fwd(phi_i,theta_i,270,d1)
	phi_2,theta_2,b_az = geod.fwd(phi_i,theta_i,270+np.rad2deg(np.arctan(10/13000.0)),d2)

	renderer = Renderer_35mm(calc,10,theta_i,phi_i,270,30000,
								vert_res=4000,focal_length=4000,vert_camera_tilt=-0.07)

	image1_path = os.path.join("images","cargo2.png")
	image2_path = os.path.join("images","iStenaLine.png")
	s.add_image(image1_path,(-1.0,theta_2,phi_2),dimensions=(-1,5))
	s.add_image(image2_path,(-0.5,theta_1,phi_1),dimensions=(-1,10))
	renderer.render_scene(s,image_name)


res=1000

# location 1 27/10/17
theta_i, phi_i = 54.487375, -3.599760

h0 = 35 

# Temperature profiles
def T_prof(h):
	e1 = np.exp(h/1.5)
	e2 = np.exp(h/0.1)
	return (2/(1+e1))*0.1+(2/(1+e2))*0.15+0.5*(np.exp(-((h-40)/3)**2)-0.0*np.exp(-(h-15)**2))

calc_args = dict(T_prof=T_prof)
calc = CurveCalc(**calc_args)

h = np.linspace(0,1000,10001)

plt.plot(calc.T(calc.R0+h)-273,h,color="blue")
plt.xlabel("Temperature (C)")
plt.ylabel("Height (m)")
plt.savefig("images/mirage/Temperature.svg")
plt.clf()

plt.plot(calc.P(calc.R0+h),h,color="green")
plt.xlabel("Pressure (Pa)")
plt.ylabel("Height (m)")
plt.savefig("images/mirage/pressure_inf.svg")
plt.clf()

plt.plot(calc.n(0,calc.R0+h),h,color="red")
plt.xlabel("index of refraction")
plt.ylabel("Height (m)")
plt.savefig("images/mirage/ior_inf.svg")
plt.clf()


plt.plot(calc.rho(calc.R0+h),h,color="orange")
plt.xlabel("density (kg/$\mathrm{m}^3$)")
plt.ylabel("Height (m)")
plt.savefig("images/mirage/density_inf.svg")
plt.clf()


h = np.linspace(0,50,1001)

plt.plot(calc.T(calc.R0+h)-273,h,color="blue")
plt.xlabel("Temperature (C)")
plt.ylabel("Height (m)")
plt.savefig("images/mirage/Temperature_inf_zoom.svg")
plt.clf()

plt.plot(calc.P(calc.R0+h),h,color="green")
plt.xlabel("Pressure (Pa)")
plt.ylabel("Height (m)")
plt.savefig("images/mirage/pressure_inf_zoom.svg")
plt.clf()

plt.plot(calc.n(0,calc.R0+h),h,color="red")
plt.xlabel("index of refraction")
plt.ylabel("Height (m)")
plt.savefig("images/mirage/ior_inf_zoom.svg")
plt.clf()


plt.plot(calc.rho(calc.R0+h),h,color="orange")
plt.xlabel("density (kg/$\mathrm{m}^3$)")
plt.ylabel("Height (m)")
plt.savefig("images/mirage/density_inf_zoom.svg")
plt.clf()


calc_args = dict(T_prof=T_prof)
calc = CurveCalc(**calc_args)
headings_mins = [251.5]
headings_maxs = [252.5]
sphere_ray_diagram_fast(calc_args,"images/mirage/ray_diagram_fata_morgana_{}.svg",theta_i,phi_i,h0,[251],100,f_length=2000,alpha_horizon=0.0)
flat_ray_diagram_fast(calc_args,"images/mirage/flat_ray_diagram_fata_morgana_{}.svg",theta_i,phi_i,h0,[251],100,f_length=2000,alpha_horizon=0.3)

sphere_image_fast(calc_args,"images/mirage/IOM_fata_morgana_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),theta_i,phi_i,h0,headings_mins,headings_maxs,res,f_length=2000,alpha_horizon=0.0)
flat_image_fast(calc_args,"images/mirage/flat_IOM_fata_morgana_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),theta_i,phi_i,h0,headings_mins,headings_maxs,res,f_length=2000,alpha_horizon=0.1)


