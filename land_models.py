import georasters as gr
import numpy as np
import scipy.interpolate as interp




class land_model(object):
    """
    This class is used to grab slices of the tarrain along great circle
    slices of the earth. 
    """
    def __init__(self,raster,R0=6371000.0,tide=0):
        NDV, xsize, ysize, GeoT, Projection, DataType = gr.get_geo_info(raster)

        self._v = GeoT[0]+np.arange(xsize)*GeoT[1]
        self._u = GeoT[3]+np.arange(ysize)*GeoT[5]

        # Load raster
        np.deg2rad(self._v,out=self._v)
        np.deg2rad(self._u,out=self._u)

        # np.mod(self._v+2*np.pi,2*np.pi,out=self._v)

        data = gr.load_tiff(raster)
        self._data = np.asarray(data,dtype=np.float32)
        self._data[self._data==np.iinfo(np.int16).min] = 0
        self._data += R0


        arg = np.argsort(self._u)
        self._data = self._data[arg,:].copy()
        self._u = self._u[arg].copy()

        self._terrain = interp.RegularGridInterpolator((self._u,self._v),self._data,bounds_error=False,fill_value=R0)
        # self._terrain = sphere_linear_interp(self._u,self._v,self._data,bounds_error=False,fill_value=R0)
        self._R0=R0
        self._tide=tide

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

    def get_sphere_pos(self,v):
        # gets vectors lats and longs
        if v.shape[0] != 3:
            raise ValueError

        x,y,z = v[0],v[1],v[2]
        theta = np.arctan2(z,np.sqrt(x**2+y**2))
        phi = np.arctan2(y,x)

        # np.mod(phi+2*np.pi,2*np.pi,out=phi)

        return theta,phi

    def rotation_matrix(self,axis, theta):
        """
        This function creates counter clockwise
        rotation matrix around axis
        """
        axis = np.asarray(axis)

        if axis.shape != (3,):
            raise ValueError

        axis = axis/np.linalg.norm(axis)
        a = np.cos(theta/2.0)
        b, c, d = -axis*np.sin(theta/2.0)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
        return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                         [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                         [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

    def slerp(self,p0,p1,n_points=10000):
        """
        this function calculates interpolation 
        on an arc between vectors p0 and p1
        """ 
        p0 = np.asarray(p0)
        if p0.shape != (3,):
            raise ValueError

        p1 = np.asarray(p1)
        if p1.shape != (3,):
            raise ValueError

        omega = np.arccos(p0.dot(p1)/(np.linalg.norm(p0)*np.linalg.norm(p1)))
        t = np.linspace(0,1,n_points,endpoint=True)

        p = (np.outer(p0,np.sin((1-t)*omega))+np.outer(p1,np.sin(omega*t)))/np.sin(omega)

        return p

    def get_sphere_pos_along_heading(self,p0,pN_start,pN_stop,heading,n_points=10000):
        # heading defined clockwise rotation from north
        p1 = self.rotation_matrix(p0,-heading).dot(pN_start)
        p2 = self.rotation_matrix(p0,-heading).dot(pN_stop)
        return  self.get_sphere_pos(self.slerp(p1,p2,n_points))

    def get_terrain_along_heading(self,theta0,phi0,heading,d_max=None,n_points=10000,d_min=0,sigmas=None):
        if sigmas is not None:
            sigma_max = sigmas.max()
            sigma_min = sigmas.min()
            n_points = len(sigmas)
        else:
            sigma_max = d_max/self._R0
            sigma_min = d_min/self._R0
            sigmas = np.linspace(sigma_min,sigma_max,n_points)

        theta0,phi0,heading = np.deg2rad([theta0,phi0,heading])

        # observer location as unit vector
        p0 = np.array([np.cos(theta0)*np.cos(phi0),
                       np.cos(theta0)*np.sin(phi0),
                       np.sin(theta0)])

        # vector at distance sigma_max in the northern headint direction.
        pN_stop = np.array([np.cos(theta0+sigma_max)*np.cos(phi0),
                       np.cos(theta0+sigma_max)*np.sin(phi0),
                       np.sin(theta0+sigma_max)])

        pN_start = np.array([np.cos(theta0+sigma_min)*np.cos(phi0),
                       np.cos(theta0+sigma_min)*np.sin(phi0),
                       np.sin(theta0+sigma_min)])

        thetas,phis = self.get_sphere_pos_along_heading(p0,pN_start,pN_stop,heading,n_points)
        heights = self._terrain(np.vstack((thetas,phis)).T)
        
        if d_min>0:
            mask = heights > self.R0
            return sigmas[mask].copy(),heights[mask].copy()+self._tide
        else:
            return sigmas,heights+self._tide

class sphere_linear_interp(object):
	def __init__(self,thetas,phis,data,method='linear',bounds_error=True, fill_value=np.nan):
		self._phis = phis
		self._y = np.arcsinh(np.tan(thetas))
		self._mercator_interp = interp.RegularGridInterpolator((self._y,self._phis),data,method=method,bounds_error=bounds_error,fill_value=fill_value)

	def __call__(self,points):
		np.tan(points[:,0],out=points[:,0])
		np.arcsinh(points[:,0],out=points[:,0])
		return self._mercator_interp(points)



