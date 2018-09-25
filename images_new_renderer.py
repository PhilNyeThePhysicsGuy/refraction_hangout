from refraction_render.calcs import CurveCalc,FlatCalc
from refraction_render.renderers import Renderer_Composite,Scene,land_model
from refraction_render.misc import mi_to_m
import numpy as np
import os,gdal,pyproj
import cProfile


def get_elevation_IOM():
	path = os.path.join("height_data","n54_w005_1arc_v3.tif")
	raster = gdal.Open(path)
	data = np.array(raster.ReadAsArray())
	n_lat,n_lon =  data.shape

	lats = np.linspace(54,55,n_lat) # get latitudes of raster
	lons = np.linspace(-5,-4,n_lon) # get longitudes of raster
	data = data[::-1,:] # data must be flipped row whys so that latitude grid is strictly increasing

	lat_max = 54.418078
	lat_min = 54.043727

	lon_max = -4.307841
	lon_min = -4.830855

	lon_mask = np.argwhere(np.logical_and(lons < lon_max,lons >= lon_min)).ravel()
	lat_mask = np.argwhere(np.logical_and(lats < lat_max,lats >= lat_min)).ravel()

	data = data[lat_mask[0]:lat_mask[-1]+1,lon_mask[0]:lon_mask[-1]+1]
	lats = lats[lat_mask].copy()
	lons = lons[lon_mask].copy()
	print data.shape

	return lats,lons,data





def make_scene():

	theta_i,phi_i=54.489822, -3.604765
	s = Scene()
	image_path = os.path.join("images","MG_lighthouse_model.png")
	s.add_image(image_path,(43.7,54.295668,-4.309418),dimensions=(-1,23))
	s.add_elevation_model(*get_elevation_IOM())

	return s


def cfunc(d,h):
	ng = 100+(255-100)*(d/45000.0)**(-4)
	nr = ng*(1-h/621.0)
	return np.stack(np.broadcast_arrays(nr,ng,0),axis=-1)



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
renderer = Renderer_Composite(calc,h0,theta_i,phi_i,mi_to_m(60),
							  vert_res=2000,focal_length=2000,distance_res=10.0,vert_obs_angle=0.1)


vfov = renderer.vfov
heading_mins = np.arange(235,259,3)
heading_maxs = heading_mins + 3
image_names = ["images/location_2/b/IOM_parts_new/IOM_location_2_b_vfov_{:.5f}_part_{}_{}_{}.png".format(vfov,h0,a,b) for a,b in zip(heading_mins,heading_maxs)]

s = make_scene()

surface_color = [23,111,197]
renderer.render_scene(s,image_names,heading_mins,heading_maxs,cfunc=cfunc,
	disp=True,eye_level=True,surface_color=surface_color)
