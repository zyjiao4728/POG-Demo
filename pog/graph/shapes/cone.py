import numpy as np
import trimesh.creation
import trimesh.visual
from pog.graph.shape import Affordance, Shape, ShapeID
import sdf
import math


class Cone(Shape):
    height: float
    radius: float

    def __init__(self,
                 shape_type=ShapeID.Cone,
                 height=0.3,
                 radius=0.08,
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
        self.export_obj()

    @classmethod
    def from_saved(cls, n: dict):
        return cls(height=n['height'],
                   radius=n['radius'],
                   transform=np.array(n['transform']))

    def create_aff(self):
        tf = self.transform @ np.array(
            ((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
        aff_nz = Affordance(name='cone_aff_nz', transform=tf, \
            shape = sdf.d2.circle(radius=self.radius), area = math.pi * self.radius**2, bb = [2.0*self.radius, 2.0*self.radius], height = self.height)

        self.add_aff(aff_nz)

    @property
    def export_file_name(self):
        return './pog_example/mesh/temp_cone_{}_{}.obj'.format(
            self.radius, self.height)

    @property
    def default_affordance_name(self) -> str:
        return 'cone_aff_nz'