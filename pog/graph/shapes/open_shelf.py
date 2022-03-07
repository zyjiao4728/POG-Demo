from ast import walk
import numpy as np
import sdf.d2
import sdf.d3
import trimesh
import trimesh.creation as creation
from pog.graph.params import WALL_THICKNESS
from pog.graph.shape import Affordance, Shape, ShapeID, ShapeType


class OpenShelf(Shape):

    def __init__(self,
                 size=np.array([0.6, 0.4, 0.4]),
                 transform=np.identity(4),
                 **kwargs):
        super().__init__(ShapeID.OpenShelf)
        size = size if isinstance(size, np.ndarray) else np.array(size)
        self.size = size
        outer_shape = creation.box(extents=size, transform=transform, **kwargs)
        inner_tf = transform - np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, WALL_THICKNESS / 2.],
            [0, 0, 0, 0],
        ])
        inner_shape = creation.box(
            extents=np.array([
                size[0] - WALL_THICKNESS, size[1] + WALL_THICKNESS,
                size[2] - WALL_THICKNESS / 2.
            ]),
            transform=inner_tf,
            **kwargs,
        )
        self.shape = outer_shape.difference(inner_shape)
        self.shape.visual.face_colors[:] = trimesh.visual.random_color()
        self.transform = transform
        self.object_type = ShapeType.ARTIC
        self.volume = self.shape.volume
        self.mass = self.volume
        self.com = np.array(self.shape.center_mass)
        self.create_aff(size)
        self.export_obj()

    @property
    def export_file_name(self):
        return './pog_example/mesh/temp_open_shelf.obj'

    def create_aff(self, size):
        outer_params = {
            "containment": False,
            "shape": sdf.d2.rectangle(size[[0, 1]]),
            "area": size[0] * size[1],
            "bb": size[[0, 1]],
            "height": size[2],
        }
        affs = [{
            "name":
            'shelf_outer_top',
            "tf":
            self.transform @ np.array((
                (1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, self.size[2] / 2.0),
                (0, 0, 0, 1),
            )),
            "params":
            outer_params,
        }, {
            "name":
            "shelf_outer_bottom",
            "tf":
            self.transform @ np.array((
                (1, 0, 0, 0),
                (0, -1, 0, 0),
                (0, 0, -1, -self.size[2] / 2.0),
                (0, 0, 0, 1),
            )),
            "params":
            outer_params,
        }]
        for aff in affs:
            self.add_aff(
                Affordance(name=aff["name"],
                           transform=aff["tf"],
                           **aff["params"]))

    @property
    def default_affordance_name(self) -> str:
        return 'shelf_outer_top'