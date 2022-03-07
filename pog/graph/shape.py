import logging
import math
import os
from abc import ABCMeta, abstractmethod
from enum import Enum

import numpy as np
import sdf
import transforms3d as tf3d
import trimesh
from pog.graph.params import MAX_STABLE_POSES, WALL_THICKNESS


class ShapeID(Enum):
    Sphere = 0
    Box = 1
    Cylinder = 2
    Cone = 3
    Storage = 4
    Imported = 5
    Wardrobe = 11
    ComplexStorage = 12
    Drawer = 13
    OpenShelf = 20


class AffordanceType(Enum):
    Support = 0


class ShapeType(Enum):
    RIGID = 0
    ARTIC = 1


class Affordance():

    def __init__(self,
                 name,
                 transform=None,
                 type=AffordanceType.Support,
                 **kwargs) -> None:
        """Initialize an affordance object

        Args:
            name (string): name of the affordance
            transform (4x4 numpy array, optional): Homogeneous transformation between parent affordance and child. Defaults to None.
            type (AffordanceType, optional): Affordance type. Defaults to AffordanceType.Support.
        """
        self.name = name
        self.node_id = None
        if transform is not None:
            # Check if transform is a valid tf matrix
            assert transform.shape == (4, 4)
            assert np.array_equal(transform[-1, :], np.array((0, 0, 0, 1)))
            self.transform = transform
        elif transform is None and type == AffordanceType.Support:
            logging.warning("Uninitialized tf, default to identity.")
            self.transform = np.array(((1, 0, 0, 0), (0, 1, 0, 0),
                                       (0, 0, 1, 0), (0, 0, 0, 1)))  # identity

        self.affordance_type = type
        self.attributes = kwargs
        self.containment = kwargs[
            'containment'] if 'containment' in kwargs.keys() else False

    def __repr__(self) -> str:
        return AffordanceType(self.affordance_type).name + ': ' + str(
            self.node_id) + ', ' + self.name

    def __eq__(self, other) -> bool:
        return self.node_id == other.node_id and self.name == other.name

    def to_lite(self):
        """Convert affordance object to lite version. Remove SDF function

        Returns:
            aff_lite (dict): A dictionary of light-weight affordance
        """
        aff_lite = {}
        aff_lite['name'] = getattr(self, 'name')
        aff_lite['node_id'] = getattr(self, 'node_id')
        aff_lite['transform'] = getattr(self, 'transform',
                                        np.identity(4).tolist())
        aff_lite['affordance_type'] = getattr(self, 'affordance_type',
                                              AffordanceType.Support)
        aff_lite['containment'] = getattr(self, 'containment', False)
        for key, value in self.attributes.items():
            if key != 'shape':
                aff_lite[key] = value
        return aff_lite

    def get_axes(self, axes='z'):
        """Get child affordance axes vector with respect to parent frame

        Args:
            axes (str, optional): Axes name. Defaults to 'z'.

        Returns:
            3x1 numpy array: axis vector
        """
        if axes == 'z':  # surface normal
            return self.transform[0:3, 2]
        elif axes == 'x':
            return self.transform[0:3, 0]
        elif axes == 'y':
            return self.transform[0:3, 1]
        else:
            logging.error(
                'Unrecognized axes. Expecting x y z, but get {}'.format(axes))


class Shape(metaclass=ABCMeta):
    shape: trimesh.Trimesh

    def __init__(self, shape_type: ShapeID) -> None:
        """Basic shape class

        Args:
            shape_type (ShapeID): shape type
        """
        self.shape_type = shape_type
        self.shape = None  # shape?
        self.transform = None  # transform from initial pose
        self.volume = None  # calculated?
        self.mass = None  # mass
        self.com = None  # center of mass
        self.affordance = {}
        self.object_type = ShapeType.RIGID

    def __repr__(self) -> str:
        return ShapeID(self.shape_type).name

    @classmethod
    @abstractmethod
    def from_saved(cls, n: dict):
        raise NotImplementedError('from_saved is not implemented for {}', cls)

    @staticmethod
    def parse_transform(global_transform):
        translation, rotation, _, _ = tf3d.affines.decompose44(
            global_transform)

        quaternion = tf3d.quaternions.mat2quat(rotation)

        if translation[-1] < 0:
            rotx180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            translation = rotx180 @ translation
            rotation = rotx180 @ rotation
            quaternion = tf3d.quaternions.mat2quat(rotation)
        return translation, quaternion

    def add_aff(self, aff: Affordance):
        self.affordance[aff.name] = aff

    def create_aff(self):
        raise NotImplementedError

    def get_aff(self):
        return self.affordance

    def gen_mesh(self, dir='out.stl'):
        self.shape.export(dir)

    def export_obj(self):
        self.shape.export(self.export_file_name)

    @property
    def default_affordance_name(self) -> str:
        """
        affordance as supportee for primitive geometries, supporter for containers
        """
        raise NotImplementedError


class Sphere(Shape):

    def __init__(self,
                 shape_type=ShapeID.Sphere,
                 radius=1.0,
                 transform=np.identity(4),
                 **kwargs) -> None:
        """Basic sphere class

        Args:
            shape_type (ShapeID, optional): shape type. Defaults to ShapeID.Sphere.
            radius (float, optional): radius of sphere. Defaults to 1.0.
            transform (4x4 numpy array, optional): transformation of this shape. Defaults to np.identity(4).
        """
        super().__init__(shape_type)
        self.radius = radius
        self.shape = trimesh.creation.icosphere(radius=radius, **kwargs)
        self.shape.visual.face_colors[:] = trimesh.visual.random_color()
        self.transform = transform
        self.shape.apply_transform(self.transform)
        self.volume = 4.0 / 3.0 * math.pi * radius**3
        self.mass = self.volume
        self.com = np.array(self.shape.center_mass)
        self.create_aff()

    def create_aff(self):
        tf = self.transform @ np.array(
            ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, self.radius), (0, 0, 0, 1)))
        aff_pz = Affordance(name='sphere_aff_pz', transform=tf, containment = False,\
            shape = sdf.d2.circle(radius=1e-1), area = 1e-1, bb = [1e-1, 1e-1], height = self.radius)

        tf = self.transform @ np.array(
            ((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, -self.radius),
             (0, 0, 0, 1)))
        aff_nz = Affordance(name='sphere_aff_nz', transform=tf, containment = False,\
            shape = sdf.d2.circle(radius=1e-1), area = 1e-1, bb = [1e-1, 1e-1], height = self.radius)

        self.add_aff(aff_pz)
        self.add_aff(aff_nz)

    @property
    def default_affordance_name(self) -> str:
        return 'sphere_aff_nz'

    @classmethod
    def from_saved(cls, n: dict):
        return cls(radius=n['radius'], transform=np.array(n['transform']))


class Box(Shape):

    def __init__(self,
                 shape_type=ShapeID.Box,
                 size=1.0,
                 transform=np.identity(4),
                 **kwargs) -> None:
        """Basic box shape

        Args:
            shape_type (ShapeID, optional): shape type. Defaults to ShapeID.Box.
            size (float or (3,) float, optional): size of box. Defaults to 1.0.
            transform (4x4 numpy array, optional): transformation of this shape. Defaults to np.identity(4).
        """
        super().__init__(shape_type)
        if isinstance(size, float) or isinstance(size, int):
            self.size = np.array([size, size, size])
        else:
            self.size = np.array(size)

        self.shape = trimesh.creation.box(extents=self.size,
                                          transform=transform,
                                          **kwargs)
        self.shape.visual.face_colors[:] = trimesh.visual.random_color()
        self.transform = transform
        self.volume = self.size[0] * self.size[1] * self.size[2]
        self.mass = self.volume
        self.com = np.array(self.shape.center_mass)
        self.create_aff()
        self.export_obj()

    @classmethod
    def from_saved(cls, n: dict):
        return cls(size=n['size'], transform=np.array(n['transform']))

    @property
    def export_file_name(self):
        return './pog_example/mesh/temp_box_{:.1f}_{:.1f}_{:.1f}.obj'.format(
            self.size[0], self.size[1], self.size[2])

    def create_aff(self):
        tf = self.transform @ np.array(
            ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, self.size[2] / 2.0),
             (0, 0, 0, 1)))
        aff_pz = Affordance(name='box_aff_pz', transform=tf, containment = False,\
            shape = sdf.d2.rectangle(self.size[[0,1]]), area = self.size[0]*self.size[1], bb = self.size[[0,1]], height = self.size[2])

        tf = self.transform @ np.array(
            ((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, -self.size[2] / 2.0),
             (0, 0, 0, 1)))
        aff_nz = Affordance(name='box_aff_nz', transform=tf, containment = False,\
            shape = sdf.d2.rectangle(self.size[[0,1]]), area = self.size[0]*self.size[1], bb = self.size[[0,1]], height = self.size[2])

        tf = self.transform @ np.array(
            ((1, 0, 0, 0), (0, 0, 1, self.size[1] / 2.0), (0, -1, 0, 0),
             (0, 0, 0, 1)))
        aff_py = Affordance(name='box_aff_py', transform=tf, containment = False,\
            shape = sdf.d2.rectangle(self.size[[0,2]]), area = self.size[0]*self.size[2], bb = self.size[[0,2]], height = self.size[1])

        tf = self.transform @ np.array(
            ((1, 0, 0, 0), (0, 0, -1, -self.size[1] / 2.0), (0, 1, 0, 0),
             (0, 0, 0, 1)))
        aff_ny = Affordance(name='box_aff_ny', transform=tf, containment = False,\
            shape = sdf.d2.rectangle(self.size[[0,2]]), area = self.size[0]*self.size[2], bb = self.size[[0,2]], height = self.size[1])

        tf = self.transform @ np.array(
            ((0, 0, 1, self.size[0] / 2.0), (0, 1, 0, 0), (-1, 0, 0, 0),
             (0, 0, 0, 1)))
        aff_px = Affordance(name='box_aff_px', transform=tf, containment = False,\
            shape = sdf.d2.rectangle(self.size[[2,1]]), area = self.size[1]*self.size[2], bb = self.size[[2,1]], height = self.size[0])

        tf = self.transform @ np.array(
            ((0, 0, -1, -self.size[0] / 2.0), (0, 1, 0, 0), (1, 0, 0, 0),
             (0, 0, 0, 1)))
        aff_nx = Affordance(name='box_aff_nx', transform=tf, containment = False,\
            shape = sdf.d2.rectangle(self.size[[2,1]]), area = self.size[1]*self.size[2], bb = self.size[[2,1]], height = self.size[0])

        self.add_aff(aff_pz)
        self.add_aff(aff_nz)
        self.add_aff(aff_py)
        self.add_aff(aff_ny)
        self.add_aff(aff_px)
        self.add_aff(aff_nx)

    @property
    def default_affordance_name(self) -> str:
        return 'box_aff_nz'


class Cylinder(Shape):

    def __init__(self,
                 shape_type=ShapeID.Cylinder,
                 height=1.0,
                 radius=1.0,
                 transform=np.identity(4),
                 **kwargs) -> None:
        """Basic cylinder shape

        Args:
            shape_type (ShapeID, optional): shape type. Defaults to ShapeID.Cylinder.
            height (float, optional): cylinder height. Defaults to 1.0.
            radius (float, optional): radius of cylinder. Defaults to 1.0.
            transform (4x4 numpy array, optional): transformation of this shape. Defaults to np.identity(4).
        """
        super().__init__(shape_type)
        self.radius = radius
        self.height = height
        self.shape = trimesh.creation.cylinder(radius=radius,
                                               height=height,
                                               transform=transform,
                                               **kwargs)
        self.shape.visual.face_colors[:] = trimesh.visual.random_color()
        self.transform = transform
        self.volume = math.pi * radius**2 * height
        self.mass = self.volume
        self.com = np.array(self.shape.center_mass)
        self.create_aff()

    @classmethod
    def from_saved(cls, n: dict):
        return cls(height=n['height'],
                   radius=n['radius'],
                   transform=np.array(n['transform']))

    def create_aff(self):
        tf = self.transform @ np.array(
            ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, self.height / 2.0),
             (0, 0, 0, 1)))
        aff_pz = Affordance(name='cylinder_aff_pz', transform=tf, containment = False,\
            shape = sdf.d2.circle(radius=self.radius), area = math.pi * self.radius**2, bb = [2.0*self.radius, 2.0*self.radius], height = self.height)

        tf = self.transform @ np.array(
            ((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, -self.height / 2.0),
             (0, 0, 0, 1)))
        aff_nz = Affordance(name='cylinder_aff_nz', transform=tf, containment = False,\
            shape = sdf.d2.circle(radius=self.radius), area = math.pi * self.radius**2, bb = [2.0*self.radius, 2.0*self.radius], height = self.height)

        self.add_aff(aff_pz)
        self.add_aff(aff_nz)
        self.export_obj()

    @property
    def default_affordance_name(self) -> str:
        return 'cylinder_aff_nz'

    @property
    def export_file_name(self):
        return './pog_example/mesh/temp_cylinder.obj'


class _Cone(Shape):

    def __init__(self,
                 shape_type=ShapeID.Cone,
                 height=1.0,
                 radius=1.0,
                 transform=np.identity(4),
                 **kwargs) -> None:
        """Basic cone shape

        Args:
            shape_type (ShapeID, optional): shape type. Defaults to ShapeID.Cone.
            height (float, optional): cone height. Defaults to 1.0.
            radius (float, optional): cone radius. Defaults to 1.0.
            transform (4x4 numpy array, optional): transformation of this shape. Defaults to np.identity(4).
        """
        super().__init__(shape_type)
        self.radius = radius
        self.height = height
        self.shape = trimesh.creation.cone(radius,
                                           height,
                                           transform=transform,
                                           **kwargs)
        self.shape.visual.face_colors[:] = trimesh.visual.random_color()
        self.transform = transform
        self.volume = self.shape.volume
        self.mass = self.volume
        self.com = np.array(self.shape.center_mass)
        self.create_aff()

    def create_aff(self):
        tf = self.transform @ np.array(
            ((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, -self.height / 2.0),
             (0, 0, 0, 1)))
        aff_nz = Affordance(name='cone_aff_nz', transform=tf, \
            shape = sdf.d2.circle(radius=self.radius), area = math.pi * self.radius**2, bb = [2.0*self.radius, 2.0*self.radius], height = self.height)

        self.add_aff(aff_nz)


# TODO: define a storage shape
class Storage(Shape):

    def __init__(self,
                 shape_type=ShapeID.Storage,
                 size=1.0,
                 transform=np.identity(4),
                 storage_type='shelf',
                 **kwargs) -> None:
        super().__init__(shape_type)

        if isinstance(size, float) or isinstance(size, int):
            self.size = np.array([size, size, size])
        else:
            self.size = np.array(size)

        outer_shape = trimesh.creation.box(extents=self.size,
                                           transform=transform,
                                           **kwargs)
        inner_tf = transform - np.array([[0, 0, 0, WALL_THICKNESS], [
            0, 0, 0, 0
        ], [0, 0, 0, 0], [0, 0, 0, 0]])
        inner_shape = trimesh.creation.box(extents=self.size - WALL_THICKNESS,
                                           transform=inner_tf,
                                           **kwargs)
        self.shape = outer_shape.difference(inner_shape)
        self.shape.visual.face_colors[:] = trimesh.visual.random_color()
        self.transform = transform
        self.object_type = ShapeType.ARTIC
        self.volume = self.shape.volume
        self.mass = self.volume
        self.com = np.array(self.shape.center_mass)
        self.create_aff(storage_type)

    @classmethod
    def from_saved(cls, n):
        return cls(size=n['size'], transform=np.array(n['transform']))

    def create_aff(self, storage_type):
        if storage_type == 'shelf':
            tf = self.transform @ np.array(
                ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, self.size[2] / 2.0),
                 (0, 0, 0, 1)))
            # storage top
            aff_pz_top = Affordance(name='shelf_aff_pz_top', transform=tf, containment = False,\
                shape = sdf.d2.rectangle(self.size[[0,1]]), area = self.size[0]*self.size[1], bb = self.size[[0,1]], height = self.size[2])

            tf = self.transform @ np.array(
                ((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, -self.size[2] / 2.0),
                 (0, 0, 0, 1)))
            # storage bottom
            aff_nz = Affordance(name='shelf_aff_nz', transform=tf, containment = False,\
                shape = sdf.d2.rectangle(self.size[[0,1]]), area = self.size[0]*self.size[1], bb = self.size[[0,1]], height = self.size[2])

            tf = self.transform @ np.array(
                ((1, 0, 0, WALL_THICKNESS), (0, 1, 0, 0),
                 (0, 0, 1, -self.size[2] / 2.0 + WALL_THICKNESS),
                 (0, 0, 0, 1)))
            # storage inner
            aff_pz_bottom = Affordance(name='shelf_aff_pz_bottom', transform=tf, containment = True, \
                shape = sdf.d2.rectangle(self.size[[0,1]]-2*WALL_THICKNESS), area = (self.size[0]-2*WALL_THICKNESS)*(self.size[1]-2*WALL_THICKNESS), bb = self.size[[0,1]]-2*WALL_THICKNESS, height = self.size[2]-2*WALL_THICKNESS)

            self.add_aff(aff_pz_top)
            self.add_aff(aff_nz)
            self.add_aff(aff_pz_bottom)

        elif storage_type == 'box':
            raise NotImplementedError(storage_type)
        else:
            logging.error('Unknown storage type: {}.'.format(type))

    @property
    def default_affordance_name(self) -> str:
        return 'shelf_aff_pz_bottom'


class Imported(Shape):

    def __init__(self,
                 shape_type=ShapeID.Imported,
                 file_path=None,
                 transform=np.identity(4),
                 **kwargs) -> None:
        """Imported shape

        Args:
            shape_type (ShapeID, optional): shape type. Defaults to ShapeID.Imported.
            file_path (str, optional): file_path to mesh. Defaults to None.
            transform (4x4 numpy array, optional): transformation of this shape. Defaults to np.identity(4).
        """
        super().__init__(shape_type)

        if file_path is not None:
            self.shape = trimesh.load(file_path, valudate=True)
            self.shape.visual.face_colors[:] = trimesh.visual.random_color()
            self.mesh_dir = file_path
        else:
            try:
                self.shape = trimesh.Trimesh(**kwargs)
                self.shape.visual.face_colors[:] = trimesh.visual.random_color(
                )
            except:
                logging.error('Unable to initialize shape: {}.'.format(
                    ShapeID.Imported.name))

        self.transform = transform
        self.shape.apply_transform(self.transform)

        center_mass = self.shape.center_mass
        self.shape.apply_transform(
            tf3d.affines.compose(-center_mass, np.identity(3), np.ones(3)))
        self.shape.export(self.mesh_dir.replace('.stl', '.obj'))

        if self.shape.is_volume:
            self.volume = self.shape.volume
        else:
            watertightness = self.shape.fill_holes()
            if watertightness:
                self.volume = self.shape.volume
            else:
                logging.warning(
                    'Use convex hull for volume approximation of incoming mesh: {}'
                    .format(file_path))
                self.volume = self.shape.convex_hull.volume

        self.mass = self.volume
        self.com = np.array(self.shape.center_mass)

        self.create_aff()

    @classmethod
    def from_saved(cls, n: dict, file_dir: str):
        try:
            return cls(file_path=n['file_path'],
                       transform=np.array(n['transform']))
        except:
            return cls(file_path=os.path.join(file_dir, 'meshes',
                                              str(n['id']) + '.stl'),
                       transform=np.array(n['transform']))

    def create_aff(self):
        tfs, _ = self.shape.compute_stable_poses()

        roty180 = np.array(
            ((-1, 0, 0, 0), (0, 1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))

        for i in range(min(MAX_STABLE_POSES, tfs.shape[0])):
            temp_shape = self.shape.copy().apply_transform(
                np.linalg.inv(np.linalg.inv(tfs[i, :, :]) @ roty180))
            bounds = np.array(temp_shape.bounds)
            extents = np.array(temp_shape.extents)
            bounds_center = (bounds[1] + bounds[0]) / 2.0
            tf = np.linalg.inv(tfs[i, :, :]) @ roty180
            self.add_aff(Affordance(name='imported_aff_{}'.format(i), transform=tf, containment = False,\
                shape = sdf.d2.rectangle(extents[[0,1]], center = np.array([-bounds_center[0], bounds_center[1]])), area = extents[0]*extents[1], bb = extents[[0,1]], height = extents[2]))

    @property
    def default_affordance_name(self) -> str:
        return 'imported_aff_0'
