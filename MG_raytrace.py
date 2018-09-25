from refraction import *
from numba import njit,prange
import georasters as gr
from PIL import Image
import matplotlib.pyplot as plt
from pyproj import Geod,pj_ellps


class land_model(object):
    """
    This class is used to grab slices of the tarrain along great circle
    slices of the earth. 
    """
    def __init__(self,ellps='WGS84',tide=0):
        NDV, xsize, ysize, GeoT, Projection, DataType = gr.get_geo_info("height_data/isle_of_man.tif")

        self._v = -4.8791666667 + np.arange(xsize)*GeoT[1]
        self._u = 54.4350000000 + np.arange(ysize)*GeoT[5]

        self._geod = Geod(ellps=ellps)

        data = gr.load_tiff("height_data/isle_of_man.tif")
        self._data = np.asarray(data,dtype=np.float32)
        self._data[self._data==np.iinfo(np.int16).min] = 0

        arg = np.argsort(self._u)
        self._data = self._data[arg,:].copy()
        self._u = self._u[arg].copy()

        lh_theta, lh_phi =  54.295718, -4.309411

        # i = np.searchsorted(self._u,lh_theta)
        # j = np.searchsorted(self._v,lh_phi)

        # self._data[i-1,j-1] = 300
        # self._data[i,j-1] = 300
        # self._data[i-1,j] = 300
        # self._data[i,j] += 300

        # bp_theta, bp_phi =  54.244236,-4.485745
        # j = np.searchsorted(self._v,bp_phi)
        # i = np.searchsorted(self._u,bp_theta)
        
        # self._data[i,j] += 200


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

    @property
    def geod(self):
    	return self._geod

    def get_terrain(self,theta0,phi0,heading,dist):
        phi,theta,az = self._geod.fwd(phi0,theta0,heading,dist)
        height = self._terrain(np.vstack((theta,phi)).T)
        return height+self._tide




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


def render_land_lighthouse(png_data,d,rs,d_min,d_max,theta_i,phi_i,headings,angles,terrain,eye_level,
						   lh_png_data,ray_height,lh_heading_min,lh_heading_max,lh_height_min,lh_height_max):
	n_v = len(angles)
	
	a_v = angles.max()-angles.min()
	i_min_angle = angles.argmin()

	hmax = terrain.data.max()

	water = np.zeros(n_v,dtype=np.bool)
	land = np.zeros(n_v,dtype=np.bool)
	sky = np.zeros(n_v,dtype=np.bool)
	lh = np.zeros(n_v,dtype=np.bool)
	inds = np.zeros(n_v,dtype=np.int32)

	azs = np.zeros_like(d)
	theta_is = np.full_like(d,theta_i)
	phi_is = np.full_like(d,phi_i)

	n_horz = lh_png_data.shape[0]
	n_vert = lh_png_data.shape[1]

	lh_x = np.linspace(lh_heading_min,lh_heading_max,n_horz)
	lh_y = np.linspace(lh_height_min,lh_height_max,n_vert)

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

		j_horz = np.searchsorted(lh_x,heading)
		if heading >= lh_heading_min and j_horz >= 0 and j_horz < n_horz:
			lh[:] = False
			not_water = np.logical_not(water)

			lh[not_water]=np.logical_and(lh_height_min <= ray_height[not_water], ray_height[not_water] < lh_height_max)
			j_vert = np.searchsorted(lh_y,ray_height[lh])

			mask = lh_png_data[j_horz,j_vert,3] > 0
			
			if np.any(mask):
				jj_vert = np.argwhere(lh).ravel()
				png_data[i,jj_vert[mask],:] = lh_png_data[j_horz,j_vert[mask],:3]
			

		print heading,headings[-1],land.sum()

def sphere_image_fast(calc_args,image_name,ellps,theta_i,phi_i,h0,heading_mins,heading_maxs,n_v=200,f_length=2000,eye_level=False,alpha_horizon=0.0):

	terrain = land_model(ellps)
	a_h,a_v = get_angle_of_view(f_length)

	calc = CurveCalc(**calc_args)

	# lighthouse image at z
	lh_image = Image.open("images/MG_lighthouse_model.png")
	lh_png_data = np.array(lh_image)

	lh_png_data = lh_png_data[::-1,:,:].transpose((1,0,2)).copy()
	aspect = float(lh_png_data.shape[0])/float(lh_png_data.shape[1])

	# lh_theta, lh_phi = 54.295850, -4.309418
	# lh_theta, lh_phi = 54.295749, -4.309419
	lh_theta, lh_phi = 54.295705, -4.309424


	lh_width = 23*aspect
	lh_height_min = 43.7
	lh_height_max = lh_height_min+23
	lh_heading,_,lh_dist = terrain.geod.inv(phi_i,theta_i,lh_phi,lh_theta)
	lh_heading = lh_heading%360
	lh_angular_rad = np.rad2deg(np.arctan2(lh_width/2.0,lh_dist))
	lh_heading_min = lh_heading - lh_angular_rad
	lh_heading_max = lh_heading + lh_angular_rad



	lats = np.kron(terrain.lats,np.ones_like(terrain.longs))
	longs = np.kron(np.ones_like(terrain.lats),terrain.longs)

	h,dists = gc_dist(theta_i,phi_i,lats,longs)

	d_min = dists.min()
	d_max = dists.max()

	d = np.arange(0,d_max,30)
	sigmas = d / terrain.R0

	a_min = alpha_horizon - a_v/2.0
	a_max = alpha_horizon + a_v/2.0

	angles,dheading = np.linspace(a_min,a_max,n_v,retstep=True)
	sols = calc.solve_ivp(d_max,h0,alpha=angles,atol=1.1e-3,rtol=1.1e-5)

	rs = sols.sol(np.pi/2+sigmas)[:n_v,:].copy()
	ray_height = sols.sol(np.pi/2+lh_dist/terrain.R0)[:n_v].copy()

	ray_height -= terrain.R0
	rs -= terrain.R0

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

		render_land_lighthouse(png_data,d,rs,d_min,d_max,theta_i,phi_i,headings,angles,
							   terrain,eye_level,lh_png_data,ray_height,lh_heading_min,lh_heading_max,lh_height_min,lh_height_max)
		

		png_data = png_data.transpose((1,0,2))
		png_data = png_data[::-1,:,:]
		im = Image.fromarray(png_data,mode="RGB")
		im.save(image_name.format(a_v,heading_min,heading_max))

	# for heading_min,heading_max in zip(heading_mins,heading_maxs):
	# 	headings = np.arange(heading_min,heading_max,dheading)
	# 	render_image_fast(d,sigmas,rs,d_min,d_max,theta_i,phi_i,headings,angles,
	# 				terrain,image_name.format(a_v,heading_min,heading_max),eye_level)

def flat_image_fast(calc_args,image_name,theta_i,phi_i,h0,heading_mins,heading_maxs,n_v=200,f_length=2000,eye_level=False,alpha_horizon=0):

	R0 = 6371000.0
	a_h,a_v = get_angle_of_view(f_length)

	calc = FlatCalc(**calc_args)


	# lighthouse image at z
	lh_image = Image.open("images/MG_lighthouse_model2.png")
	lh_png_data = np.array(lh_image)

	# lh_png_data = lh_png_data[::-1,:,:].transpose((1,0,2)).copy()
	lh_png_data = lh_png_data.transpose((1,0,2)).copy()
	aspect = float(lh_png_data.shape[0])/float(lh_png_data.shape[1])

	lh_theta, lh_phi = 54.295668, -4.309418
	lh_width = 23*aspect
	lh_height_min = 43.7
	lh_height_max = lh_height_min+23
	lh_heading,_,lh_dist = terrain.geod.inv(phi_i,theta_i,lh_phi,lh_theta)
	lh_heading = lh_heading%360
	lh_angular_rad = np.arctan2(lh_width/2.0,lh_dist)
	lh_heading_min = np.rad2deg(lh_heading - lh_angular_rad)
	lh_heading_max = np.rad2deg(lh_heading + lh_angular_rad)

	terrain = land_model("WGS84")

	lh_png_data = lh_png_data[::-1,:,:].transpose((1,0,2)).copy()
	aspect = float(lh_png_data.shape[0])/float(lh_png_data.shape[1])

	lats = np.kron(terrain.lats,np.ones_like(terrain.longs))
	longs = np.kron(np.ones_like(terrain.lats),terrain.longs)


	h,dists = gc_dist(theta_i,phi_i,lats,longs)

	d_min = dists.min()
	d_max = dists.max()

	d = np.arange(0,d_max,30)
	sigmas = d/R0

	a_min = alpha_horizon - a_v/2.0
	a_max = alpha_horizon + a_v/2.0

	angles,dheading = np.linspace(a_min,a_max,n_v,retstep=True)
	sols = calc.solve_ivp(d_max,h0,alpha=angles,atol=1.1e-3,rtol=1.1e-5)

	rs = sols.sol(d)[:n_v,:].copy()
	ray_height = sols.sol(lh_dist)[:n_v].copy()

	lh_png_data = lh_png_data[::-1,:,:].transpose((1,0,2)).copy()

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

		render_land_lighthouse(png_data,d,rs,d_min,d_max,theta_i,phi_i,headings,angles,
							   terrain,eye_level,lh_png_data,ray_height,lh_heading_min,lh_heading_max,lh_height_min,lh_height_max)
		

		png_data = png_data.transpose((1,0,2))
		png_data = png_data[::-1,:,:]
		im = Image.fromarray(png_data,mode="RGB")
		im.save(image_name.format(a_v,heading_min,heading_max))

	# for heading_min,heading_max in zip(heading_mins,heading_maxs):
	# 	headings = np.arange(heading_min,heading_max,dheading)
	# 	render_image_fast(d,sigmas,rs,d_min,d_max,theta_i,phi_i,headings,angles,
	# 				terrain,image_name.format(a_v,heading_min,heading_max),eye_level)



res=4000

# location 1 27/10/17
theta_i, phi_i = 54.487375, -3.599760


h0 = 35 

# Temperature profiles
def T_prof(h):
	e1 = np.exp(h/1.5)
	e2 = np.exp(h/0.1)
	return (2/(1+e1))*0.1+(2/(1+e2))*0.15

calc_args = dict(T0=8.3,P0=103000,T_prof=T_prof)
sphere_image_fast(calc_args,"Full_Render_shift_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"sphere",theta_i,phi_i,h0,235,259,500,f_length=2000,alpha_horizon=0.05,eye_level=True)

# sphere_image_fast(calc_args,"images/location_1/MG_lighthouse/sphere/MG_lighthouse_looming_mirage_WGS84_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"WGS84",theta_i,phi_i,h0,245,246,res,f_length=2000,alpha_horizon=0.1,eye_level=True)
# sphere_image_fast(calc_args,"images/location_1/MG_lighthouse/sphere/MG_lighthouse_looming_mirage_WGS84_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"WGS84",theta_i,phi_i,h0,245.4,245.5,res,f_length=16000,alpha_horizon=-0.15)
# sphere_image_fast(calc_args,"images/location_1/MG_lighthouse/sphere/MG_lighthouse_looming_mirage_sphere_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"sphere",theta_i,phi_i,h0,245,246,res,f_length=2000,alpha_horizon=0.1,eye_level=True)
# sphere_image_fast(calc_args,"images/location_1/MG_lighthouse/sphere/MG_lighthouse_looming_mirage_sphere_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"sphere",theta_i,phi_i,h0,245.35,245.45,res,f_length=16000,alpha_horizon=-0.15)



calc_args = dict(T0=8.3,P0=103000)
# sphere_image_fast(calc_args,"images/location_1/MG_lighthouse/sphere/MG_lighthouse_looming_WGS84_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"WGS84",theta_i,phi_i,h0,245,246,res,f_length=2000,alpha_horizon=0.1,eye_level=True)
# sphere_image_fast(calc_args,"images/location_1/MG_lighthouse/sphere/MG_lighthouse_looming_WGS84_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"WGS84",theta_i,phi_i,h0,245.4,245.5,res,f_length=16000,alpha_horizon=-0.15)
# sphere_image_fast(calc_args,"images/location_1/MG_lighthouse/sphere/MG_lighthouse_looming_sphere_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"sphere",theta_i,phi_i,h0,245,246,res,f_length=2000,alpha_horizon=0.1,eye_level=True)
# sphere_image_fast(calc_args,"images/location_1/MG_lighthouse/sphere/MG_lighthouse_looming_sphere_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"sphere",theta_i,phi_i,h0,245.35,245.45,res,f_length=16000,alpha_horizon=-0.15)



calc_args = dict(n_funcs=(lambda t,r:1.0,lambda t,r:0.0,lambda t,r:0.0))
# sphere_image_fast(calc_args,"images/location_1/MG_lighthouse/sphere/MG_lighthouse_no_ref_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"sphere",theta_i,phi_i,h0,245,246,res,f_length=2000,alpha_horizon=0.1,eye_level=True)
# sphere_image_fast(calc_args,"images/location_1/MG_lighthouse/sphere/MG_lighthouse_no_ref_WGS84_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"WGS84",theta_i,phi_i,h0,245,246,res,f_length=2000,alpha_horizon=0.1,eye_level=True)
# sphere_image_fast(calc_args,"images/location_1/MG_lighthouse/sphere/MG_lighthouse_no_ref_WGS84_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),"WGS84",theta_i,phi_i,h0,245.35,245.45,res,f_length=16000,alpha_horizon=-0.15)



# calc_args = dict(T0=8.3,P0=103000,T_prof=T_prof)
# flat_image_fast(calc_args,"images/location_1/MG_lighthouse/flat/MG_lighthouse_looming_mirage_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),theta_i,phi_i,h0,245,246,res,f_length=2000,alpha_horizon=0.3,eye_level=True)
# flat_image_fast(calc_args,"images/location_1/MG_lighthouse/flat/MG_lighthouse_looming_mirage_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),theta_i,phi_i,h0,245.35,245.45,res,f_length=16000,alpha_horizon=0.05)

# calc_args = dict(T0=8.3,P0=103000)
# flat_image_fast(calc_args,"images/location_1/MG_lighthouse/flat/MG_lighthouse_looming_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),theta_i,phi_i,h0,245,246,res,f_length=2000,alpha_horizon=0.3,eye_level=True)
# flat_image_fast(calc_args,"images/location_1/MG_lighthouse/flat/MG_lighthouse_looming_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),theta_i,phi_i,h0,245.35,245.45,res,f_length=16000,alpha_horizon=0.05)

# calc_args = dict(n_funcs=(lambda t,r:1.0,lambda t,r:0.0,lambda t,r:0.0))
# flat_image_fast(calc_args,"images/location_1/MG_lighthouse/flat/MG_lighthouse_no_ref_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),theta_i,phi_i,h0,245,246,res,f_length=2000,alpha_horizon=0.3,eye_level=True)
# flat_image_fast(calc_args,"images/location_1/MG_lighthouse/flat/MG_lighthouse_no_ref_vfov_{{:.5f}}_part_{}_{{}}_{{}}.png".format(h0),theta_i,phi_i,h0,245.35,245.45,res,f_length=16000,alpha_horizon=0.05)
