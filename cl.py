"""
romiscan.cl
___________
This module contains all OpenCL accelerated functions.
"""
import os
import numpy as np
import pyopencl as cl
import glob
#from proc3d import point2index
from skimage.morphology import binary_dilation
import cv2
import json

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

prg_dir = os.path.join(os.path.dirname(__file__), 'kernels')

eps = 1e-10

with open(os.path.join(prg_dir, 'backprojection.c')) as f:
    backprojection_kernels = cl.Program(
        ctx, f.read()).build(options="-I%s" % prg_dir)


class Backprojection(object):

    def __init__(self, shape, origin, voxel_size):
        """
        Parameters
        ----------
        shape: list
            Shape of the voxel volume.
        origin: list
            Location of the origin of the voxel space.
        voxel_size: float
            Size of voxels.
        """
        self.shape = shape
        self.origin = origin
        self.voxel_size = voxel_size
        self.default_value = 0
        self.dtype = np.int32
        self.kernel = backprojection_kernels.carve
        #self.dtype = np.float32
        #self.kernel = backprojection_kernels.average
        buff_size = np.prod(self.shape) * 4
        # Defines attributes used to initialize OpenCL buffers:
        self.values_h = None
        self.values_d = None
        self.intrinsics_d = None
        self.rot_d = None
        self.tvec_d = None
        self.volinfo_d = None
        self.shape_d = None
        # Set attributes values for OpenCL buffers:
        self.init_buffers()

    def init_buffers(self):
        """Helper function to initialize OpenCL buffers."""
        self.values_h = self.default_value * \
                        np.ones(self.shape, dtype=self.dtype)

        self.values_d = cl.Buffer(
            ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.values_h)

        self.intrinsics_d = cl.Buffer(
            ctx, mf.READ_ONLY, np.zeros(4, dtype=np.float32).nbytes)
        self.rot_d = cl.Buffer(
            ctx, mf.READ_ONLY, np.zeros(9, dtype=np.float32).nbytes)
        self.tvec_d = cl.Buffer(
            ctx, mf.READ_ONLY, np.zeros(3, dtype=np.float32).nbytes)

        self.volinfo_d = cl.Buffer(
            ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.array(
                [*self.origin, self.voxel_size], dtype=np.float32)
        )

        self.shape_d = cl.Buffer(
            ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.array(
                self.shape, dtype=np.int32)
        )

    def process_view(self, intrinsics, rot, tvec, mask):
        """Process a new view.
        Parameters
        ----------
        intrinsics: list
            [f_x, f_y, c_x, c_y]
        rot: list of list
            rotation matrix of the camera pose
        tvec: list
            translation vector of the camera pose
        mask: np.ndarray
            mask array (or float array if type is averaging)
        """
        if self.dtype == np.float32 and mask.dtype == np.uint8:
            mask = mask / 255
        
        intrinsics_h = np.ascontiguousarray(intrinsics).astype(np.float32)
        rot_h = np.ascontiguousarray(rot).astype(np.float32)
        tvec_h = np.ascontiguousarray(tvec).astype(np.float32)
        mask_h = np.ascontiguousarray(mask).astype(self.dtype)

        mask_d = cl.image_from_array(ctx, mask_h, 1)
        cl.enqueue_copy(queue, self.intrinsics_d, intrinsics_h)
        cl.enqueue_copy(queue, self.rot_d, rot_h)
        cl.enqueue_copy(queue, self.tvec_d, tvec_h)
                
        self.kernel(queue, [np.prod(self.shape)], None, mask_d, self.values_d,
                    self.intrinsics_d, self.rot_d,
                    self.tvec_d, self.volinfo_d, self.shape_d)
        queue.finish()

    def values(self):
        """Gets computed values from the OpenCL device."""
        cl.enqueue_copy(queue, self.values_h, self.values_d)
        return self.values_h

    def process_fileset(self, folder, n_dilation):
        """Processes a whole fileset for given label.
        Parameters
        ----------
        fs : romidata.DB.Fileset
            Fileset to process.
        """
        self.clear()

        files=glob.glob(folder+"/images/*")
        files.sort()
        
        for i,fi in enumerate(files):
            #print("processing frame %s of %s"%(i, len(files)))
            name = os.path.basename(fi)[:-4]
            cam = json.load(open(folder+"/metadata/images/"+name+".json"))
            camera_model = cam["camera_model"]
            width = camera_model['width']
            height = camera_model['height']
            intrinsics = camera_model['params'][0:4]

            #rot = [item for sublist in cam['R']  for item in sublist]
            #rot = sum(cam['rotmat'], [])
            rot = sum(cam['R'], [])
            #tvec = cam['tvec']
            tvec = cam['T']
            mask = cv2.imread(fi,0)
            if n_dilation:
               for i in range(n_dilation): mask = binary_dilation(mask)    

            self.process_view(intrinsics, rot, tvec, mask)

        result = self.values()
        result = result.reshape(self.shape)
        return result

    def clear(self):
        """Clear computed values from the OpenCL device."""
        self.values_h = self.default_value * \
                        np.ones(self.shape).astype(self.dtype)
        cl.enqueue_copy(queue, self.values_d, self.values_h)
