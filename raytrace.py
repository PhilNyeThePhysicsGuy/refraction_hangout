import georasters as gr
import scipy.interpolate as interp
import scipy.optimize as op
import numpy as np
import matplotlib.pyplot as plt
from numba import njit,jit
import PIL.Image as Image
from refraction import *
import os,sys
import cProfile
from memory_profiler import profile



@njit
def ray_crossing(rs,heights,water,land,sky,inds):
	n_v = rs.shape[0]
	m = rs.shape[1]

	water[:] = False
	land[:] = False
	sky[:] = False
	inds[:] = -1

	for i in range(n_v):
		hit = False
		for j in range(m):
			if rs[i,j] <= heights[j]:
				hit = True
				if heights[j] > 0.01:
					land[i] = True
					inds[i] = j
				else:
					water[i] = True
					inds[i] = j

				break

		sky[i] = not hit

def render_image_fast(d,sigmas,rs,d_min,d_max,theta_i,phi_i,headings,angles,terrain,image_name,eye_level=False):
	n_v = len(angles)
	
	R0 = 6371000.0
	a_v = angles.max()-angles.min()
	i_min_angle = angles.argmin()

	hmax = float(terrain.data.max()-R0)

	data_name = image_name.replace(".png",".npy")

	if os.path.isfile(data_name):
		print "png data exists, re-rendering from existing data."
		png_data = np.load(data_name)
		im = Image.fromarray(png_data,mode="RGB")
		im.save(image_name)
		return

	png_data = np.empty((len(headings),len(angles),3),dtype=np.uint8)
	png_data[...] = 0

	water = np.zeros(n_v,dtype=np.bool)
	land = np.zeros(n_v,dtype=np.bool)
	sky = np.zeros(n_v,dtype=np.bool)
	inds = np.zeros(n_v,dtype=np.int32)

	if eye_level:
		i_horz = np.abs(angles)<(a_v/400.0)
	else:
		i_horz = np.array([])


	for i,heading in enumerate(headings):
		
		sigmas,heights = terrain.get_terrain_along_heading(theta_i,phi_i,heading,sigmas=sigmas)
		heights -= R0

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


		print heading,headings[-1],land.sum()
		if np.any(i_horz):
			png_data[i,i_horz,0] = 255
			png_data[i,i_horz,1] = 100
			png_data[i,i_horz,2] = 0

	png_data = png_data.transpose((1,0,2))
	png_data = png_data[::-1,:,:]
	im = Image.fromarray(png_data,mode="RGB")
	im.save(image_name)


def sphere_image_rectilinear_fast(calc_args,image_name,h0,h_pos,v_pos,n_v=200,f_length=2000,eye_level=False):
	global phi_i,theta_i

	R0 = 6371000.0
	a_h,a_v = get_angle_of_view(f_length)

	calc = CurveCalc(**calc_args)
	new_calc_args = dict(calc_args)
	try:
		new_calc_args.pop("T_prof")
		calc_nm = CurveCalc(**new_calc_args)
	except KeyError:
		calc_nm = calc

	terrain = land_model("height_data/isle_of_man.tif")

	lats = np.kron(np.rad2deg(terrain.lats),np.ones_like(terrain.longs))
	longs = np.kron(np.ones_like(terrain.lats),np.rad2deg(terrain.longs))

	h,dists = gc_dist(theta_i,phi_i,lats,longs)

	d_min = dists.min()
	d_max = dists.max()

	d = np.arange(0,d_max,10)
	sigmas = d / R0

	h_angles,v_angles = get_angles_of_view(f_length,n_v)

	h_pos = np.array(h_pos)
	v_pos = np.array(v_pos)

	if v_pos.ndim==0:
		v_pos = v_pos.reshape((1,))

	if h_pos.ndim==0:
		h_pos = h_pos.reshape((1,))

	for v_mid in v_pos:
		angles = v_angles+v_mid
		sols = calc.solve_ivp(d_max,h0,alpha=angles,atol=1.1e-6,rtol=1.1e-6)
		rs = sols.sol(np.pi/2+sigmas)[:n_v,:].copy()
		rs -= R0
		for h_mid in h_pos:
			headings = h_angles+h_mid
			render_image_fast(d,sigmas,rs,d_min,d_max,theta_i,phi_i,headings,
				angles,terrain,image_name.format(h_mid,v_mid),eye_level)

def sphere_image_fast(calc_args,image_name,h0,heading_mins,heading_maxs,n_v=200,f_length=2000,eye_level=False,alpha_horizon=None):

	global phi_i,theta_i

	R0 = 6371000.0
	a_h,a_v = get_angle_of_view(f_length)

	calc = CurveCalc(**calc_args)
	new_calc_args = dict(calc_args)
	try:
		new_calc_args.pop("T_prof")
		calc_nm = CurveCalc(**new_calc_args)
	except KeyError:
		calc_nm = calc

	terrain = land_model("height_data/isle_of_man.tif")

	lats = np.kron(np.rad2deg(terrain.lats),np.ones_like(terrain.longs))
	longs = np.kron(np.ones_like(terrain.lats),np.rad2deg(terrain.longs))

	h,dists = gc_dist(theta_i,phi_i,lats,longs)

	d_min = dists.min()
	d_max = dists.max()



	d = np.arange(0,d_max,10)
	sigmas = d / R0

	if alpha_horizon is None:
		dh,k,h_sol=calc_nm.solve_hidden(d_max,h0)
		alpha_horizon = np.rad2deg(-polar_angle(h_sol.sol,np.pi/2))
		a_min = alpha_horizon - a_v/2.0
		a_max = alpha_horizon + a_v/2.0
	else:
		a_min = alpha_horizon - a_v/2.0
		a_max = alpha_horizon + a_v/2.0

	angles,dangle = np.linspace(a_min,a_max,n_v,retstep=True)
	sols = calc.solve_ivp(d_max,h0,alpha=angles,atol=1.1e-6,rtol=1.1e-6)

	rs = sols.sol(np.pi/2+sigmas)[:n_v,:].copy()
	rs -= R0

	dheading = angles[1]-angles[0]

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
					terrain,image_name.format(heading_min,heading_max),eye_level)

def flat_image_fast(calc_args,image_name,h0,heading_mins,heading_maxs,n_v=200,f_length=2000,eye_level=False,alpha_horizon=0):
	global phi_i,theta_i

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

	angles = np.linspace(a_min,a_max,n_v)
	sols = calc.solve_ivp(d_max,h0,alpha=angles,atol=1.1e-3,rtol=1.1e-5)

	rs = sols.sol(d)[:n_v,:].copy()

	dheading = angles[1]-angles[0]

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
					terrain,image_name.format(heading_min,heading_max),eye_level)

def sphere_ray_diagram_fast(calc_args,image_name,h0,headings,n_v=200,f_length=2000,eye_level=False,alpha_horizon=0):
	global phi_i,theta_i

	R0 = 6371000.0
	a_h,a_v = get_angle_of_view(f_length)

	calc = CurveCalc(**calc_args)
	try:
		calc_args.pop("T_prof")
		calc_nm = CurveCalc(**calc_args)
	except KeyError:
		calc_nm = calc

	terrain = land_model("height_data/isle_of_man.tif")

	lats = np.kron(np.rad2deg(terrain.lats),np.ones_like(terrain.longs))
	longs = np.kron(np.ones_like(terrain.lats),np.rad2deg(terrain.longs))

	h,dists = gc_dist(theta_i,phi_i,lats,longs)

	d_min = dists.min()
	d_max = dists.max()

	d = np.arange(0,d_max,10)
	sigmas = d / R0

	if alpha_horizon is None:
		dh,k,h_sol=calc_nm.solve_hidden(d_max,h0)
		alpha_horizon = np.rad2deg(-polar_angle(h_sol.sol,np.pi/2))
		a_min = alpha_horizon - a_v/2.0
		a_max = alpha_horizon + a_v/2.0
	else:
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

		for i in range(n_v):
			if i == i_horz and eye_level:
				i_max = inds[i]
				plt.plot(rs[i,:i_max]*c[:i_max],
					     rs[i,:i_max]*s[:i_max],color="orange",linewidth=0.2)
				continue

			if water[i]:
				i_max = inds[i]
				plt.plot(rs[i,:i_max]*c[:i_max],
					     rs[i,:i_max]*s[:i_max],color="blue",linewidth=0.2)
				continue

			if land[i]:
				i_max = inds[i]
				plt.plot(rs[i,:i_max]*c[:i_max],
					     rs[i,:i_max]*s[:i_max],color="green",linewidth=0.2)
				continue

			if sky[i]:
				plt.plot(rs[i,:]*c,rs[i,:]*s,color="cyan",linewidth=0.2)
				continue

		plt.plot(c*heights,s*heights,color="green",linewidth=0.2)
		plt.plot(c*R0,s*R0,color="blue",linewidth=0.2)
		ax = plt.gca()
		ylim = ax.get_ylim()
		xlim = ax.get_xlim()
		plt.xticks([])
		plt.yticks([])
		plt.savefig(image_name.format(heading))
		plt.clf()

def flat_ray_diagram_fast(calc_args,image_name,h0,headings,n_v=200,f_length=2000,eye_level=False,alpha_horizon=0,save_opt={}):
	global phi_i,theta_i

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
	sigmas = d / R0

	a_min = alpha_horizon - a_v/2.0
	a_max = alpha_horizon + a_v/2.0

	angles = np.linspace(a_min,a_max,n_v)
	sols = calc.solve_ivp(d_max,h0,alpha=angles,atol=1.1e-5,rtol=1.1e-5)

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
				plt.plot(d[:i_max],rs[i,:i_max],color="orange",linewidth=0.2)
				continue

			if water[i]:
				i_max = inds[i]
				plt.plot(d[:i_max],rs[i,:i_max],color="blue",linewidth=0.2)
				continue

			if land[i]:
				i_max = inds[i]
				plt.plot(d[:i_max],rs[i,:i_max],color="green",linewidth=0.2)
				continue

			if sky[i]:
				plt.plot(d,rs[i,:],color="cyan",linewidth=0.2)
				continue

		plt.plot(d,heights,color="green",linewidth=0.2)
		plt.plot(d,np.zeros_like(d),color="blue",linewidth=0.2)
		ax = plt.gca()
		ylim = ax.get_ylim()
		xlim = ax.get_xlim()
		plt.xticks([])
		plt.yticks([])
		plt.savefig(image_name.format(heading),**save_opt)
		plt.clf()


def sphere_ray_dist_animation(calc_args,image_name,h0,h1,distances,n_v=200,f_length=2000,eye_level=False,alpha_horizon=0,save_opt={}):
	R0 = 6371000.0

	
	a_h,a_v = get_angle_of_view(f_length)

	calc = CurveCalc(**calc_args)
	try:
		calc_args.pop("T_prof")
		calc_nm = CurveCalc(**calc_args)
	except KeyError:
		calc_nm = calc

	distances = np.asarray(distances,dtype=np.float64)
	d_max = max(distances)+200
	d_min = min(distances)

	d = np.arange(0,d_max,10)
	sigmas = d/R0

	if alpha_horizon is None:
		dh,k,h_sol=calc_nm.solve_hidden(d_max,h0)
		alpha_horizon = np.rad2deg(-polar_angle(h_sol.sol,np.pi/2))
		a_min = alpha_horizon - a_v/2.0
		a_max = alpha_horizon + a_v/2.0
	else:
		a_min = alpha_horizon - a_v/2.0
		a_max = alpha_horizon + a_v/2.0

	angles = np.linspace(a_min,a_max,n_v)
	sols = calc.solve_ivp(d_max,h0,alpha=angles,atol=1.1e-3,rtol=1.1e-5)

	water = np.zeros(n_v,dtype=np.bool)
	land = np.zeros(n_v,dtype=np.bool)
	sky = np.zeros(n_v,dtype=np.bool)
	inds = np.zeros(n_v,dtype=np.int32)

	heights = np.zeros_like(d)

	rs = sols.sol(np.pi/2+sigmas)[:n_v,:].copy()
	c = np.cos(np.pi/2+sigmas)
	s = np.sin(np.pi/2+sigmas)

	rs -= R0
	ray_crossing(rs,heights,water,land,sky,inds)
	rs += R0
	heights += R0

	i_horz = np.abs(angles).argmin()

	for i in range(n_v):
		if i == i_horz and eye_level:
			i_max = inds[i]
			plt.plot(rs[i,:i_max]*c[:i_max],
				     rs[i,:i_max]*s[:i_max]-R0,color="orange",linewidth=0.2)
			continue

		if water[i]:
			i_max = inds[i]
			plt.plot(rs[i,:i_max]*c[:i_max],
				     rs[i,:i_max]*s[:i_max]-R0,color="blue",linewidth=0.2)
			continue

		if land[i]:
			i_max = inds[i]
			plt.plot(rs[i,:i_max]*c[:i_max],
				     rs[i,:i_max]*s[:i_max]-R0,color="green",linewidth=0.2)
			continue

		if sky[i]:
			plt.plot(rs[i,:]*c,rs[i,:]*s-R0,color="cyan",linewidth=0.2)
			continue

	ymax = 3*h1
	plt.plot(c*heights,s*heights-R0,color="green",linewidth=0.2)
	plt.plot(c*R0,s*R0-R0,color="blue",linewidth=0.2)
	ax = plt.gca()
	ylim = ax.get_ylim()
	xlim = ax.get_xlim()
	plt.xticks([])
	plt.yticks([])
	plt.clf()

	ylim = list(ylim)
	ylim[-1] = ymax


	for distance in distances:
		heights[:] = 0.0
		mask = np.logical_and(d>=distance,d<(distance+200))
		heights[mask] = h1
		
		rs -= R0
		ray_crossing(rs,heights,water,land,sky,inds)
		rs += R0
		heights += R0

		i_horz = np.abs(angles).argmin()

		for i in range(n_v):
			if i == i_horz and eye_level:
				i_max = inds[i]
				plt.plot(rs[i,:i_max]*c[:i_max],
					     rs[i,:i_max]*s[:i_max]-R0,color="orange",linewidth=0.2)
				continue

			if water[i]:
				i_max = inds[i]
				plt.plot(rs[i,:i_max]*c[:i_max],
					     rs[i,:i_max]*s[:i_max]-R0,color="blue",linewidth=0.2)
				continue

			if land[i]:
				i_max = inds[i]
				plt.plot(rs[i,:i_max]*c[:i_max],
					     rs[i,:i_max]*s[:i_max]-R0,color="green",linewidth=0.2)
				continue

			if sky[i]:
				plt.plot(rs[i,:]*c,rs[i,:]*s-R0,color="cyan",linewidth=0.2)
				continue

		plt.plot(c*heights,s*heights-R0,color="green",linewidth=0.2)
		plt.plot(c*R0,s*R0-R0,color="blue",linewidth=0.2)
		ax = plt.gca()
		ax.set_ylim(ylim)
		ax.set_xlim(xlim)
		plt.xticks([])
		plt.yticks([])
		plt.title("distance: {} km obs height: {} m \n target height: {} m".format(distance/1000,h0,h1),fontsize=18)
		plt.savefig(image_name.format(distance))
		plt.clf()

def flat_ray_dist_animation(calc_args,image_name,h0,h1,distances,n_v=200,f_length=2000,eye_level=False,alpha_horizon=0,save_opt={}):

	
	a_h,a_v = get_angle_of_view(f_length)

	calc = FlatCalc(**calc_args)

	distances = np.asarray(distances,dtype=np.float64)
	d_max = max(distances)+200
	d_min = min(distances)

	d = np.arange(0,d_max,10)

	if alpha_horizon is None:
		alpha_horizon = 0.0

	a_min = alpha_horizon - a_v/2.0
	a_max = alpha_horizon + a_v/2.0

	angles = np.linspace(a_min,a_max,n_v)
	sols = calc.solve_ivp(d_max,h0,alpha=angles,atol=1.1e-3,rtol=1.1e-5)

	water = np.zeros(n_v,dtype=np.bool)
	land = np.zeros(n_v,dtype=np.bool)
	sky = np.zeros(n_v,dtype=np.bool)
	inds = np.zeros(n_v,dtype=np.int32)

	heights = np.zeros_like(d)

	i_horz = np.abs(angles).argmin()
	ylim = (0,1.3*h0)
	xlim = (0,d_max)

	for distance in distances:
		heights[:] = 0.0
		mask = np.logical_and(d>=distance,d<(distance+200))
		heights[mask] = h1
		
		ray_crossing(rs,heights,water,land,sky,inds)

		i_horz = np.abs(angles).argmin()

		for i in range(n_v):
			if i == i_horz and eye_level:
				i_max = inds[i]
				plt.plot(d_max-d[:i_max],rs[i,:i_max],color="orange",linewidth=0.2)
				continue

			if water[i]:
				i_max = inds[i]
				plt.plot(d_max-d[:i_max],rs[i,:i_max],color="blue",linewidth=0.2)
				continue

			if land[i]:
				i_max = inds[i]
				plt.plot(d_max-d[:i_max],rs[i,:i_max],color="green",linewidth=0.2)
				continue

			if sky[i]:
				plt.plot(d_max-d,rs[i,:],color="cyan",linewidth=0.2)
				continue

		plt.plot(d_max-d,heights,color="green",linewidth=0.2)
		plt.plot(d_max-d,np.zeros_like(d),color="blue",linewidth=0.2)
		ax = plt.gca()
		ax.set_ylim(ylim)
		ax.set_xlim(xlim)
		plt.xticks([])
		plt.yticks([])
		plt.savefig(image_name.format(distance))
		plt.clf()

def sphere_building_dist_animation(calc_args,image_name,h0,h1,w1,distances,n_v=200,f_length=2000,eye_level=False,alpha_horizon=0,save_opt={}):
	R0 = 6371000.0

	a_h,a_v = get_angle_of_view(f_length)

	calc = CurveCalc(**calc_args)
	try:
		copy_calc_args = dict(calc_args)
		copy_calc_args.pop("T_prof")
		calc_nm = CurveCalc(**new_calc_args)
	except KeyError:
		calc_nm = calc

	n_h = int(n_v*(a_h/a_v))
	headings = np.linspace(-a_h/2.0,a_h/2.0,n_h)

	distances = np.asarray(distances,dtype=np.float64)
	d_max = max(distances)+20
	d_min = min(distances)

	d = np.arange(0,d_max,10)

	sigmas = d/R0

	if alpha_horizon is None:
		dh,k,h_sol=calc_nm.solve_hidden(d_max,h0)
		alpha_horizon = np.rad2deg(-polar_angle(h_sol.sol,np.pi/2))
		a_min = alpha_horizon - a_v/2.0
		a_max = alpha_horizon + a_v/2.0
	else:
		a_min = alpha_horizon - a_v/2.0
		a_max = alpha_horizon + a_v/2.0

	angles = np.linspace(a_min,a_max,n_v)
	sols = calc.solve_ivp(d_max,h0,alpha=angles,atol=1.1e-3,rtol=1.1e-5)

	water = np.zeros(n_v,dtype=np.bool)
	land = np.zeros(n_v,dtype=np.bool)
	sky = np.zeros(n_v,dtype=np.bool)
	inds = np.zeros(n_v,dtype=np.int32)

	heights = np.zeros_like(d)

	rs = sols.sol(np.pi/2+sigmas)[:n_v,:].copy()
	rs -= R0

	if eye_level:
		i_horz = np.abs(angles)<(a_v/200.0)
	else:
		i_horz = np.array([])

	
	for distance in distances:
		print distance
		png_data = np.zeros((n_h,n_v,3),dtype=np.uint8)
		mask = np.logical_and(d>=distance,d<(distance+30))

		for i,heading in enumerate(headings):
			heights[:] = 0.0
			x = min(max(distance*np.tan(np.deg2rad(heading)),-w1),w1)
			h = (h1/float(w1))*np.sqrt(w1**2-x**2).real
			heights[mask] = h


			ray_crossing(rs,heights,water,land,sky,inds)
			png_data[i,water,2] = 160
			png_data[i,sky,1] = 180
			png_data[i,sky,2] = 255
			if np.any(land):
				land_inds = inds[land]
				png_data[i,land,1] = 255*(rs[land,land_inds]/h1)**0.5



			# print heading,headings[-1],land.sum()
			if np.any(i_horz):
				png_data[i,i_horz,0] = 255
				png_data[i,i_horz,1] = 100
				png_data[i,i_horz,2] = 0


		png_data = png_data.transpose((1,0,2))
		png_data = png_data[::-1,:,:]
		im = Image.fromarray(png_data,mode="RGB")
		im.save(image_name.format(distance))			

def flat_building_dist_animation(calc_args,image_name,h0,h1,w1,distances,n_v=200,f_length=2000,eye_level=False,alpha_horizon=0,save_opt={}):

	a_h,a_v = get_angle_of_view(f_length)

	calc = FlatCalc(**calc_args)


	n_h = int(n_v*(a_h/a_v))
	headings = np.linspace(-a_h/2.0,a_h/2.0,n_h)

	distances = np.asarray(distances,dtype=np.float64)
	d_max = max(distances)+20
	d_min = min(distances)

	d = np.arange(0,d_max,10)

	if alpha_horizon is None:
		alpha_horizon = 0.0

	a_min = alpha_horizon - a_v/2.0
	a_max = alpha_horizon + a_v/2.0

	angles = np.linspace(a_min,a_max,n_v)
	sols = calc.solve_ivp(d_max,h0,alpha=angles,atol=1.1e-3,rtol=1.1e-5)

	water = np.zeros(n_v,dtype=np.bool)
	land = np.zeros(n_v,dtype=np.bool)
	sky = np.zeros(n_v,dtype=np.bool)
	inds = np.zeros(n_v,dtype=np.int32)

	heights = np.zeros_like(d)

	rs = sols.sol(d)[:n_v,:].copy()

	if eye_level:
		i_horz = np.abs(angles)<(a_v/200.0)
	else:
		i_horz = np.array([])

	
	for distance in distances:
		print distance
		png_data = np.zeros((n_h,n_v,3),dtype=np.uint8)
		mask = np.logical_and(d>=distance,d<(distance+30))

		for i,heading in enumerate(headings):
			heights[:] = 0.0
			x = min(max(distance*np.tan(np.deg2rad(heading)),-w1),w1)
			h = (h1/float(w1))*np.sqrt(w1**2-x**2).real
			heights[mask] = h


			ray_crossing(rs,heights,water,land,sky,inds)
			png_data[i,water,2] = 160
			png_data[i,sky,1] = 180
			png_data[i,sky,2] = 255
			if np.any(land):
				land_inds = inds[land]
				png_data[i,land,1] = 255*(rs[land,land_inds]/h1)**0.5



			# print heading,headings[-1],land.sum()
			if np.any(i_horz):
				png_data[i,i_horz,0] = 255
				png_data[i,i_horz,1] = 100
				png_data[i,i_horz,2] = 0


		png_data = png_data.transpose((1,0,2))
		png_data = png_data[::-1,:,:]
		im = Image.fromarray(png_data,mode="RGB")
		im.save(image_name.format(distance))			



def sphere_ray_height_animation(calc_args,image_name,h0s,h1,distance,n_v=200,f_length=2000,eye_level=False,alpha_horizon=0,save_opt={}):
	R0 = 6371000.0

	a_h,a_v = get_angle_of_view(f_length)

	calc = CurveCalc(**calc_args)
	try:
		copy_calc_args = dict(calc_args)
		copy_calc_args.pop("T_prof")
		calc_nm = CurveCalc(**copy_calc_args)
	except KeyError:
		calc_nm = calc

	d_max = distance+200
	d = np.arange(0.0,distance*1.25,10.0)

	sigmas = d/R0

	if alpha_horizon is None:
		dh,k,h_sol=calc_nm.solve_hidden(d_max,h0)
		alpha_horizon = np.rad2deg(-polar_angle(h_sol.sol,np.pi/2))
		a_min = alpha_horizon - a_v/2.0
		a_max = alpha_horizon + a_v/2.0
	else:
		a_min = alpha_horizon - a_v/2.0
		a_max = alpha_horizon + a_v/2.0

	angles = np.linspace(a_min,a_max,n_v)

	water = np.zeros(n_v,dtype=np.bool)
	land = np.zeros(n_v,dtype=np.bool)
	sky = np.zeros(n_v,dtype=np.bool)
	inds = np.zeros(n_v,dtype=np.int32)

	heights = np.zeros_like(d)
	i_horz = np.abs(angles).argmin()
	heights[:] = 0.0
	mask = np.logical_and(d>=distance,d<(distance+200))
	heights[mask] = h1
	heights += R0

	h0 = h0s[-1]

	sols = calc.solve_ivp(d_max,h0,alpha=angles,atol=1.1e-3,rtol=1.1e-5)

	rs = sols.sol(np.pi/2+sigmas)[:n_v,:].copy()
	c = np.cos(np.pi/2+sigmas)
	s = np.sin(np.pi/2+sigmas)

	heights -= R0
	rs -= R0
	ray_crossing(rs,heights,water,land,sky,inds)
	rs += R0
	heights += R0

	for i in range(n_v):
		if i == i_horz and eye_level:
			i_max = inds[i]
			plt.plot(rs[i,:i_max]*c[:i_max],
				     rs[i,:i_max]*s[:i_max]-R0,color="orange",linewidth=0.2)
			continue

		if water[i]:
			i_max = inds[i]
			plt.plot(rs[i,:i_max]*c[:i_max],
				     rs[i,:i_max]*s[:i_max]-R0,color="blue",linewidth=0.2)
			continue

		if land[i]:
			i_max = inds[i]
			plt.plot(rs[i,:i_max]*c[:i_max],
				     rs[i,:i_max]*s[:i_max]-R0,color="green",linewidth=0.2)
			continue

		if sky[i]:
			plt.plot(rs[i,:]*c,rs[i,:]*s-R0,color="cyan",linewidth=0.2)
			continue

	ymax = 3*h1
	plt.plot(c*heights,s*heights-R0,color="green",linewidth=0.2)
	plt.plot(c*R0,s*R0-R0,color="blue",linewidth=0.2)
	ax = plt.gca()
	ylim = ax.get_ylim()
	xlim = ax.get_xlim()
	plt.xticks([])
	plt.yticks([])
	plt.clf()

	ylim = list(ylim)
	ylim[-1] = ymax

	for h0 in h0s:

		sols = calc.solve_ivp(d_max,h0,alpha=angles,atol=1.1e-3,rtol=1.1e-5)

		rs = sols.sol(np.pi/2+sigmas)[:n_v,:].copy()

		heights -= R0
		rs -= R0
		ray_crossing(rs,heights,water,land,sky,inds)
		rs += R0
		heights += R0

		for i in range(n_v):
			if i == i_horz and eye_level:
				i_max = inds[i]
				plt.plot(rs[i,:i_max]*c[:i_max],
					     rs[i,:i_max]*s[:i_max]-R0,color="orange",linewidth=0.2)
				continue

			if water[i]:
				i_max = inds[i]
				plt.plot(rs[i,:i_max]*c[:i_max],
					     rs[i,:i_max]*s[:i_max]-R0,color="blue",linewidth=0.2)
				continue

			if land[i]:
				i_max = inds[i]
				plt.plot(rs[i,:i_max]*c[:i_max],
					     rs[i,:i_max]*s[:i_max]-R0,color="green",linewidth=0.2)
				continue

			if sky[i]:
				plt.plot(rs[i,:]*c,rs[i,:]*s-R0,color="cyan",linewidth=0.2)
				continue

		plt.plot(c*heights,s*heights-R0,color="green",linewidth=0.2)
		plt.plot(c*R0,s*R0-R0,color="blue",linewidth=0.2)
		ax = plt.gca()
		ax.set_ylim(ylim)
		ax.set_xlim(xlim)
		plt.xticks([])
		plt.yticks([])
		plt.title("distance: {} km obs height: {} m \n target height: {} m".format(distance/1000,h0,h1),fontsize=18)
		plt.savefig(image_name.format(h0))
		plt.clf()

def flat_ray_height_animation(calc_args,image_name,h0s,h1,distance,n_v=200,f_length=2000,eye_level=False,alpha_horizon=0,save_opt={}):
	
	a_h,a_v = get_angle_of_view(f_length)

	calc = FlatCalc(**calc_args)

	d_max = distance+200
	d = np.arange(0.0,distance*1.25,10.0)

	if alpha_horizon is None:
		alpha_horizon = 0.0

	a_min = alpha_horizon - a_v/2.0
	a_max = alpha_horizon + a_v/2.0

	angles = np.linspace(a_min,a_max,n_v)

	water = np.zeros(n_v,dtype=np.bool)
	land = np.zeros(n_v,dtype=np.bool)
	sky = np.zeros(n_v,dtype=np.bool)
	inds = np.zeros(n_v,dtype=np.int32)

	heights = np.zeros_like(d)
	i_horz = np.abs(angles).argmin()
	heights[:] = 0.0
	mask = np.logical_and(d>=distance,d<(distance+200))
	heights[mask] = h1

	h0 = h0s[-1]

	ylim = (0,3*h1)
	xlim = (0,d.max())

	for h0 in h0s:

		sols = calc.solve_ivp(d.max(),h0,alpha=angles,atol=1.1e-3,rtol=1.1e-5)

		rs = sols.sol(d)[:n_v,:].copy()

		ray_crossing(rs,heights,water,land,sky,inds)

		for i in range(n_v):
			if i == i_horz and eye_level:
				i_max = inds[i]
				plt.plot(d.max()-d[:i_max],rs[i,:i_max],color="orange",linewidth=0.2)
				continue

			if water[i]:
				i_max = inds[i]
				plt.plot(d.max()-d[:i_max],rs[i,:i_max],color="blue",linewidth=0.2)
				continue

			if land[i]:
				i_max = inds[i]
				plt.plot(d.max()-d[:i_max],rs[i,:i_max],color="green",linewidth=0.2)
				continue

			if sky[i]:
				plt.plot(d.max()-d,rs[i,:],color="cyan",linewidth=0.2)
				continue

		plt.plot(d.max()-d,heights,color="green",linewidth=0.2)
		plt.plot(d.max()-d,np.zeros_like(d),color="blue",linewidth=0.2)
		ax = plt.gca()
		ax.set_ylim(ylim)
		ax.set_xlim(xlim)
		plt.xticks([])
		plt.yticks([])
		plt.savefig(image_name.format(h0))
		plt.clf()

def sphere_building_height_animation(calc_args,image_name,h0s,h1,w1,distance,n_v=200,f_length=2000,eye_level=False,alpha_horizon=0):
	R0 = 6371000.0

	a_h,a_v = get_angle_of_view(f_length)

	calc = CurveCalc(**calc_args)
	try:
		calc_args.pop("T_prof")
		calc_nm = CurveCalc(**calc_args)
	except KeyError:
		calc_nm = calc

	n_h = int(n_v*(a_h/a_v))
	headings = np.linspace(-a_h/2.0,a_h/2.0,n_h)

	d_max = distance+20

	d = np.arange(0,distance*1.25,10)

	sigmas = d/R0

	if alpha_horizon is None:
		dh,k,h_sol=calc_nm.solve_hidden(d_max,h0)
		alpha_horizon = np.rad2deg(-polar_angle(h_sol.sol,np.pi/2))
		a_min = alpha_horizon - a_v/2.0
		a_max = alpha_horizon + a_v/2.0
	else:
		a_min = alpha_horizon - a_v/2.0
		a_max = alpha_horizon + a_v/2.0

	angles = np.linspace(a_min,a_max,n_v)

	water = np.zeros(n_v,dtype=np.bool)
	land = np.zeros(n_v,dtype=np.bool)
	sky = np.zeros(n_v,dtype=np.bool)
	inds = np.zeros(n_v,dtype=np.int32)

	heights = np.zeros_like(d)

	if eye_level:
		i_horz = np.abs(angles)<(a_v/200.0)
	else:
		i_horz = np.array([])

	mask = np.logical_and(d>=distance,d<(distance+30))

	for h0 in h0s:

		png_data = np.zeros((n_h,n_v,3),dtype=np.uint8)
		sols = calc.solve_ivp(d.max(),h0,alpha=angles,atol=1.1e-3,rtol=1.1e-5)
		rs = sols.sol(np.pi/2+sigmas)[:n_v,:].copy()
		rs -= R0

		for i,heading in enumerate(headings):
			heights[:] = 0.0
			x = min(max(distance*np.tan(np.deg2rad(heading)),-w1),w1)
			h = (h1/float(w1))*np.sqrt(w1**2-x**2).real
			heights[mask] = h


			ray_crossing(rs,heights,water,land,sky,inds)
			png_data[i,water,2] = 160
			png_data[i,sky,1] = 180
			png_data[i,sky,2] = 255
			if np.any(land):
				land_inds = inds[land]
				png_data[i,land,1] = 255*(rs[land,land_inds]/h1)**0.5



			# print heading,headings[-1],land.sum()
			if np.any(i_horz):
				png_data[i,i_horz,0] = 255
				png_data[i,i_horz,1] = 100
				png_data[i,i_horz,2] = 0


		png_data = png_data.transpose((1,0,2))
		png_data = png_data[::-1,:,:]
		im = Image.fromarray(png_data,mode="RGB")
		im.save(image_name.format(h0))		

def flat_building_height_animation(calc_args,image_name,h0s,h1,w1,distance,n_v=200,f_length=2000,eye_level=False,alpha_horizon=0):

	a_h,a_v = get_angle_of_view(f_length)

	calc = FlatCalc(**calc_args)

	n_h = int(n_v*(a_h/a_v))
	headings = np.linspace(-a_h/2.0,a_h/2.0,n_h)

	d_max = distance+20

	d = np.arange(0,distance*4.0,10)

	if alpha_horizon is None:
		alpha_horizon = 0.0

	a_min = alpha_horizon - a_v/2.0
	a_max = alpha_horizon + a_v/2.0

	angles = np.linspace(a_min,a_max,n_v)

	water = np.zeros(n_v,dtype=np.bool)
	land = np.zeros(n_v,dtype=np.bool)
	sky = np.zeros(n_v,dtype=np.bool)
	inds = np.zeros(n_v,dtype=np.int32)

	heights = np.zeros_like(d)

	if eye_level:
		i_horz = np.abs(angles)<(a_v/200.0)
	else:
		i_horz = np.array([])

	mask = np.logical_and(d>=distance,d<(distance+30))

	for h0 in h0s:

		png_data = np.zeros((n_h,n_v,3),dtype=np.uint8)
		sols = calc.solve_ivp(d.max(),h0,alpha=angles,atol=1.1e-3,rtol=1.1e-5)
		rs = sols.sol(d)[:n_v,:].copy()

		for i,heading in enumerate(headings):
			heights[:] = 0.0
			x = min(max(distance*np.tan(np.deg2rad(heading)),-w1),w1)
			h = (h1/float(w1))*np.sqrt(w1**2-x**2).real
			heights[mask] = h

			ray_crossing(rs,heights,water,land,sky,inds)
			png_data[i,water,2] = 160
			png_data[i,sky,1] = 180
			png_data[i,sky,2] = 255
			if np.any(land):
				land_inds = inds[land]
				png_data[i,land,1] = 255*(rs[land,land_inds]/h1)**0.5



			# print heading,headings[-1],land.sum()
			if np.any(i_horz):
				png_data[i,i_horz,0] = 255
				png_data[i,i_horz,1] = 100
				png_data[i,i_horz,2] = 0


		png_data = png_data.transpose((1,0,2))
		png_data = png_data[::-1,:,:]
		im = Image.fromarray(png_data,mode="RGB")
		im.save(image_name.format(h0))


# older code below

if __name__ == '__main__':

	# theta_i, phi_i = 54.487375, -3.599760
	# theta_i,phi_i= 54.490301,-3.608629

	# floaty land
	# heading_min = 255.0
	# heading_max = 259.0

	# nose cone
	# heading_min = 251.5
	# heading_max = 252.5

	# whole island
	heading_min = 235.0
	heading_max = 259.0

	# 17:39 April 4, 2018 obs
	#------------------
	# calc_args = dict(T0=2.2,h0=124,P0=93000,moist_lapse_rate=True)
	# sphere_image("IOM_april_4_2018.png",13,255.0,259.0,200,f_length=3000,eye_level=True,**calc_args)

	# 17:39 April 21, 2018 obs
	# ------------------

	def smooth_f(x,a=1):
		return np.abs((x+a*np.logaddexp(x/a,-x/a))/2.0)
	# Temperature model
	def T_prof(h):
		return A*np.exp(-smooth_f(h/a)**b)

	# #temperature data
	# h_p = np.array([0.2,1,49,106,124])
	# T_p = np.array([8.7,10.2,14.8,13.5,13.1])
	# dT,A,a,b = (0.022718052655596625, -11.316761651287964, 2.048298133841327, 0.8637633318607845)
	# calc_args = dict(T0=13.1,P0=100600,h0=124,dT=dT,T_prof=T_prof)


	# sphere_rays(calc_args,0.3,257,250,f_length=256000,alpha_horizon=0.2105)
	# sphere_image(calc_args,"floaty_super_loom_1_0.3m.png",0.3,255.0,259.0,500,f_length=8000,alpha_horizon=0.2105)
	# sphere_image(calc_args,"floaty_super_loom_1_0.3m.png",0.3,256.5,257.5,500,f_length=64000,alpha_horizon=0.2105)

	# h_p = np.array([0.2,1,106,124])
	# T_p = np.array([8.7,10.2,13.5,13.1])
	# dT,A,a,b = (0.022545485639136054, -14.260344642831658, 1.3824337491371654, 0.4438666476146248)
	# calc_args = dict(T0=13.1,P0=100600,h0=124,dT=dT,T_prof=T_prof)
	# calc_args = dict(T0=13.1,P0=100600,h0=124,dT=dT)


	# sphere_rays(calc_args,0.3,257,500,f_length=64000,alpha_horizon=0.2)
	# sphere_image(calc_args,"floaty_super_loom_2_0.3m.png",0.3,256.5,257.5,2000,f_length=2000,alpha_horizon=0.2)
	# sphere_image_fast(calc_args,"test_0.3.png",0.3,255,259,1000,f_length=8000,alpha_horizon=0.2)

	# dh = 4
	# heading_mins = np.arange(235,259,dh)

	# sphere_image_fast(calc_args,"IOM_parts_35/IOM_35_{}_{}.png",35,heading_mins,heading_mins+1,4000,f_length=2000,alpha_horizon=0.1,eye_level=True)
	# sphere_image_fast(calc_args,"IOM_parts_2/IOM_2_{}_{}.png",2,heading_mins,heading_mins+1,4000,f_length=2000,alpha_horizon=0.3,eye_level=True)
	# sphere_image_fast(calc_args,"IOM_2_{}_{}.png",2,235,259,200,f_length=2000,alpha_horizon=0.3,eye_level=True)


	# heading_mins = np.arange(254.5,259,0.5)
	# sphere_image_fast(calc_args,"IOM_parts_2/IOM_2_4x_zoom_northern_part_{}_{}.png",2,heading_mins,heading_mins+0.5,4000,f_length=8000,alpha_horizon=0.13,eye_level=True)
	# sphere_ray_diagram_fast(calc_args,"IOM_parts_2/sphere_rays_IOM_2_{}.svg",2,[251,257],500,f_length=8000,alpha_horizon=0.13)
	# flat_ray_diagram_fast(calc_args,"IOM_parts_2/flat_IOM_rays_2_{}.svg",2,[251,257],500,f_length=2000,alpha_horizon=0.26)
	# flat_image_fast(calc_args,"IOM_parts_2/flat_IOM_2_{}_{}.png",2,heading_mins,heading_mins+dh,1000,f_length=2000,alpha_horizon=0.3,eye_level=True)



	# dT,A,a,b = 0.0098,-11.5, 1.22,0.5 # 0
	# dT,A,a,b = 0.0098,-5.5, 1.2,0.5 # 1
	# dT,A,a,b = 0.0098,-5.5, 3.2,0.5 # 2
	# dT,A,a,b = 0.023, -11.3, 2.0, 0.86 # 3
	# dT,A,a,b = 0.0098,-10, 3.0,0.5 # 4
	# dT,A,a,b = 0.0098,-10, 3.0,1.0 # 5
	# dT,A,a,b = 0.0098,-5, 3.0,1.0 # 6
	# dT,A,a,b = 0.0098,-5, 2.0,1.0 # 7

	# dT,A,a,b = 0.0098, -5.5, 4.0,0.5 # 9
	# dT,A,a,b = 0.0098, -11, 3.5,0.5 # 10

	# h_p_bouy = np.array([1])
	# T_p_bouy = np.array([10])

	# h_p = np.array([49,106,124])
	# T_p = np.array([16.3,15.8,15.2])


	# dT,A,a,b = 0.0098,-11.3, 4.0,0.5 # 8
	# calc_args = dict(T0=14.5,P0=100600,h0=124,dT=dT,T_prof=T_prof)

	# calc = CurveCalc(**calc_args)
	# h = np.linspace(0,150,1001)
	# plt.plot(calc.T(calc.R0+h)-273,h,color="blue")
	# plt.plot(T_p_bouy,h_p_bouy,marker=".",linestyle="",color="green")
	# plt.plot(T_p-1,h_p,marker=".",linestyle="",color="green")
	# plt.plot(T_p,h_p,marker=".",linestyle="",color="red")
	# plt.xlabel("Temperature (C)")
	# plt.ylabel("Height (m)")
	# plt.figure()

	# plt.plot(calc.n(0,calc.R0+h),h,color="red")
	# plt.xlabel("index of refraction")
	# plt.ylabel("Height (m)")
	# plt.show()

	# dT,A,a,b = 0.0098,-11.3, 4.0,0.5 # 8
	# calc_args = dict(T0=14.5,P0=100600,h0=124,dT=dT)

	# sphere_image_fast({},"IOM_test_8.1_{}_{{}}_{{}}.png".format(35),35,235,259,200,f_length=2000,alpha_horizon=0.0,eye_level=True)


	h0=3
	# sphere_image_fast(calc_args,"IOM_full_test_8.1_{}_{{}}_{{}}.png".format(h0),h0,246,253,300,f_length=2000,alpha_horizon=0.11,eye_level=True)
	# headings = np.arange(235,259,2)
	# sphere_image_fast(calc_args,"IOM_full_8.1_{}_{{}}_{{}}.png".format(h0),h0,headings,headings+2,2000,f_length=3000,alpha_horizon=0.21,eye_level=True)

	# headings = np.arange(251,259,1)
	# sphere_image_fast(calc_args,"IOM_north_test_8.1_{}_{{}}_{{}}.png".format(h0),h0,headings,headings+1,500,f_length=16000,alpha_horizon=0.13)
	# sphere_image_fast(calc_args,"IOM_test_8.1_{}_{{}}_{{}}.png".format(h0),h0,headings,headings+1,500,f_length=2000,alpha_horizon=0.3,eye_level=True)
	# flat_image_fast(calc_args,"IOM_flat_north_test_8.1_{}_{{}}_{{}}.png".format(h0),h0,251,259,500,f_length=2000,alpha_horizon=0.4,eye_level=True)
	# flat_image_fast(calc_args,"IOM_flat_test_8.1_{}_{{}}_{{}}.png".format(h0),h0,246,253,500,f_length=2000,alpha_horizon=0.4,eye_level=True)


	# flat_ray_diagram_fast(calc_args,"flat_IOM_rays_3_{}.svg",3,[251,257],500,f_length=2000,alpha_horizon=0.26)
	# sphere_ray_diagram_fast(calc_args,"sphere_rays_IOM_no_land_{}.svg",3,[100],500,f_length=128000,alpha_horizon=0.117)
	# sphere_ray_diagram_fast(calc_args,"sphere_rays_IOM_3_{}.svg",3,[251],100,f_length=10000,alpha_horizon=0.12)


	# sphere_ray_diagram_fast({},"sphere_rays_IOM_3_standard_cond_{}.svg",3,[251,257],100,f_length=4000,alpha_horizon=-0.1)
	# sphere_ray_diagram_fast({},"sphere_rays_IOM_35_standard_cond_{}.svg",35,[251,257],100,f_length=4000,alpha_horizon=-0.1)

	# sphere_ray_diagram_fast({},"sphere_rays_IOM_3_standard_cond_{}.svg",3,[251,257],100,f_length=4000,alpha_horizon=-0.1)
	# sphere_ray_diagram_fast({},"sphere_rays_IOM_35_standard_cond_{}.svg",35,[251,257],100,f_length=4000,alpha_horizon=-0.1)

	# def T_prof(h):
	# 	return (2/(1+np.exp(h/1.5)))*0.1+(2/(1+np.exp(h/0.1)))*0.15

	# h0 = 3
	# calc_args = dict(T0=8.3,P0=103000,T_prof=T_prof)
	# sphere_image_fast(calc_args,"IOM_apr21_pos_oct27_condition_inv_{}_{{}}_{{}}.png".format(h0),h0,246,253,500,f_length=2000,alpha_horizon=0.13,eye_level=True)
	# calc_args = dict(T0=8.3,P0=103000)
	# sphere_image_fast(calc_args,"IOM_apr21_pos_oct27_condition_{}_{{}}_{{}}.png".format(h0),h0,246,253,500,f_length=2000,alpha_horizon=0.13,eye_level=True)

	# h0 = 35
	# calc_args = dict(T0=8.3,P0=103000,T_prof=T_prof)
	# sphere_image_fast(calc_args,"IOM_apr21_pos_apr21_condition_inv_{}_{{}}_{{}}.png".format(h0),h0,246,253,500,f_length=2000,alpha_horizon=0.13,eye_level=True)
	# calc_args = dict(T0=8.3,P0=103000)
	# sphere_image_fast(calc_args,"IOM_apr21_pos_apr21_condition_{}_{{}}_{{}}.png".format(h0),h0,246,253,500,f_length=2000,alpha_horizon=0.13,eye_level=True)

	# theta_i, phi_i = 54.487375, -3.599760

	# h0 = 3
	# calc_args = dict(T0=8.3,P0=103000,T_prof=T_prof)
	# sphere_image_fast(calc_args,"IOM_oct27_pos_oct27_condition_inv_{}_{{}}_{{}}.png".format(h0),h0,246,253,500,f_length=2000,alpha_horizon=0.13,eye_level=True)
	# calc_args = dict(T0=8.3,P0=103000)
	# sphere_image_fast(calc_args,"IOM_oct27_pos_oct27_condition_{}_{{}}_{{}}.png".format(h0),h0,246,253,500,f_length=2000,alpha_horizon=0.13,eye_level=True)

	# h0 = 35
	# calc_args = dict(T0=8.3,P0=103000,T_prof=T_prof)
	# sphere_image_fast(calc_args,"IOM_oct27_pos_oct27_condition_inv_{}_{{}}_{{}}.png".format(h0),h0,246,253,500,f_length=2000,alpha_horizon=0.13,eye_level=True)
	# calc_args = dict(T0=8.3,P0=103000)
	# sphere_image_fast(calc_args,"IOM_oct27_pos_oct27_condition_{}_{{}}_{{}}.png".format(h0),h0,246,253,500,f_length=2000,alpha_horizon=0.13,eye_level=True)




	# GIFS

	# calc_args = dict(n_funcs=(lambda t,r:1.0,lambda t,r:0.0,lambda t,r:0.0))
	# distances = np.arange(3000,30000,500)

	# sphere_ray_dist_animation(calc_args,"gifs/dist_gif_3m_no_ref/sphere_rays_animation_{}.png",3,20,distances,f_length=2000,alpha_horizon=0.1,save_opt=dict(dpi=400))
	# flat_ray_dist_animation(calc_args,"gifs/dist_gif_3m_no_ref/flat_rays_animation_{}.png",3,20,distances,f_length=2000,alpha_horizon=0.1,save_opt=dict(dpi=400))
	# sphere_building_dist_animation(calc_args,"gifs/dist_gif_3m_no_ref/sphere_building_animation_{}.png",3,20,10,distances,n_v=500,f_length=2000,alpha_horizon=0.1)
	# flat_building_dist_animation(calc_args,"gifs/dist_gif_3m_no_ref/flat_building_animation_{}.png",3,20,10,distances,n_v=500,f_length=2000,alpha_horizon=0.1)

	# sphere_ray_dist_animation(calc_args,"gifs/dist_gif_10m_no_ref/sphere_rays_animation_{}.png",3,20,distances,f_length=2000,alpha_horizon=0.1,save_opt=dict(dpi=400))
	# flat_ray_dist_animation(calc_args,"gifs/dist_gif_10m_no_ref/flat_rays_animation_{}.png",3,20,distances,f_length=2000,alpha_horizon=0.1,save_opt=dict(dpi=400))
	# sphere_building_dist_animation(calc_args,"gifs/dist_gif_10m_no_ref/sphere_building_animation_{}.png",3,20,10,distances,n_v=500,f_length=2000,alpha_horizon=0.1)
	# flat_building_dist_animation(calc_args,"gifs/dist_gif_10m_no_ref/flat_building_animation_{}.png",3,20,10,distances,n_v=500,f_length=2000,alpha_horizon=0.1)


	# h0s = np.arange(2,51,1)
	# sphere_ray_height_animation(calc_args,"gifs/height_gif_30km_no_ref/sphere_rays_animation_{}.png",h0s,40,30000,f_length=2000,alpha_horizon=-0.1,save_opt=dict(dpi=400))
	# flat_ray_height_animation(calc_args,"gifs/height_gif_30km_no_ref/flat_rays_animation_{}.png",h0s,40,30000,f_length=2000,alpha_horizon=-0.0,save_opt=dict(dpi=400))
	# sphere_building_height_animation(calc_args,"gifs/height_gif_30km_no_ref/sphere_building_animation_{}.png",h0s,40,100,30000,n_v=500,f_length=2000,alpha_horizon=-0.1)
	# flat_building_height_animation(calc_args,"gifs/height_gif_30km_no_ref/flat_building_animation_{}.png",h0s,40,100,30000,n_v=500,f_length=2000,alpha_horizon=-0.0)



	# def T_prof(h):
	#  	return (2/(1+np.exp(h/1.5)))*0.1+(2/(1+np.exp(h/0.1)))*0.15

	# calc_args = dict(T_prof=T_prof)

	# sphere_ray_height_animation(calc_args,"gifs/height_gif_30km_inf/sphere_rays_animation_{}.png",h0s,40,30000,f_length=2000,alpha_horizon=-0.0,save_opt=dict(dpi=400))
	# flat_ray_height_animation(calc_args,"gifs/height_gif_30km_inf/flat_rays_animation_{}.png",h0s,40,30000,f_length=2000,alpha_horizon=-0.0,save_opt=dict(dpi=400))
	# sphere_building_height_animation(calc_args,"gifs/height_gif_30km_inf/sphere_building_animation_{}.png",h0s,40,100,30000,n_v=500,f_length=2000,alpha_horizon=-0.0)
	# flat_building_height_animation(calc_args,"gifs/height_gif_30km_inf/flat_building_animation_{}.png",h0s,40,100,30000,n_v=500,f_length=2000,alpha_horizon=-0.0)

	# h0s = np.arange(2,51,1)
	# sphere_ray_height_animation({},"gifs/height_gif_30km_std_ref/sphere_rays_animation_{}.png",h0s,40,30000,f_length=2000,alpha_horizon=-0.0,save_opt=dict(dpi=400))
	# flat_ray_height_animation({},"gifs/height_gif_30km_std_ref/flat_rays_animation_{}.png",h0s,40,30000,f_length=2000,alpha_horizon=-0.0,save_opt=dict(dpi=400))
	# sphere_building_height_animation({},"gifs/height_gif_30km_std_ref/sphere_building_animation_{}.png",h0s,40,100,30000,n_v=500,f_length=2000,alpha_horizon=-0.0)
	# flat_building_height_animation({},"gifs/height_gif_30km_std_ref/flat_building_animation_{}.png",h0s,40,100,30000,n_v=500,f_length=2000,alpha_horizon=-0.0)


	def T_prof(h):
		return 0.5*np.exp(-((h-20)/3)**2)

	calc_args = dict(T_prof=T_prof)
	h0s = np.arange(2,71,1)
	# sphere_ray_height_animation(calc_args,"gifs/height_gif_30km_sup/sphere_rays_animation_{}.png",h0s,40,30000,f_length=2000,alpha_horizon=-0.0,save_opt=dict(dpi=400))
	# flat_ray_height_animation(calc_args,"gifs/height_gif_30km_sup/flat_rays_animation_{}.png",h0s,40,30000,f_length=2000,alpha_horizon=-0.0,save_opt=dict(dpi=400))
	# sphere_building_height_animation(calc_args,"gifs/height_gif_30km_sup/sphere_building_animation_{}.png",h0s,40,100,30000,n_v=500,f_length=2000,alpha_horizon=-0.0)
	flat_building_height_animation(calc_args,"gifs/height_gif_30km_sup/flat_building_animation_{}.png",h0s,40,100,30000,n_v=500,f_length=2000,alpha_horizon=-0.0)



	# def T_prof(h):
	# 	return (2/(1+np.exp(h/1)))*1.0

	# h0=20
	# calc_args = dict(T_prof=T_prof)
	# sphere_image_fast(calc_args,"high_mirage_{}_{{}}_{{}}.png".format(h0),h0,251,252,500,f_length=2000,alpha_horizon=-0.1,eye_level=True)

	# def T_prof(h):
	# 	return (2/(1+np.exp(h/1)))*0.5

	# calc_args = dict(T_prof=T_prof)
	# sphere_image_fast(calc_args,"med_mirage_{}_{{}}_{{}}.png".format(h0),h0,251,252,500,f_length=2000,alpha_horizon=-0.1,eye_level=True)

	# def T_prof(h):
	# 	return (2/(1+np.exp(h/1)))*0.1

	# calc_args = dict(T_prof=T_prof)
	# sphere_image_fast(calc_args,"low_mirage_{}_{{}}_{{}}.png".format(h0),h0,251,252,500,f_length=2000,alpha_horizon=-0.1,eye_level=True)
	# sphere_image_fast({},"no_mirage_{}_{{}}_{{}}.png".format(h0),h0,251,252,500,f_length=2000,alpha_horizon=-0.1,eye_level=True)






