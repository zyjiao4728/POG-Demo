from array import ArrayType

import numpy as np
import pybullet as p
import sdf.d2
import sdf.d3
import trimesh
import trimesh.creation as creation
import trimesh.visual
from pog.graph.params import BULLET_GROUND_OFFSET, WALL_THICKNESS
from pog.graph.shape import Affordance, Shape, ShapeID, ShapeType


class Drawer(Shape):
    size: ArrayType

    def __init__(self,
                 shape_type=ShapeID.Drawer,
                 size=np.array([0.8, 0.8, 0.4]),
                 transform=np.identity(4),
                 storage_type='drawer',
                 **kwargs):
        super().__init__(shape_type)
        if shape_type not in [ShapeID.Drawer]:
            raise Exception('inwalid shape type of drawer')
        size = np.array(size)
        self.size = size
        self.transform = transform

        self.object_type = ShapeType.ARTIC
        outer_shape = creation.box(size, transform, **kwargs)
        inner_tf = transform - np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, -WALL_THICKNESS],
            [0, 0, 0, 0],
        ])
        inner_shape = creation.box(
            extents=size - np.array(WALL_THICKNESS),
            transform=inner_tf,
            **kwargs,
        )

        self.shape: trimesh.Trimesh = outer_shape.difference(inner_shape)
        self.volume = self.shape.volume
        self.mass = self.volume
        self.shape.visual.face_colors[:] = trimesh.visual.random_color()
        self.com = np.array(self.shape.center_mass)
        self.create_aff(storage_type, size)

    @classmethod
    def from_saved(cls, n: dict):
        return cls(size=n['size'], transform=np.array(n['transform']))

    def create_aff(self, storage_type: str, size):
        outer_params = {
            "containment": False,
            "shape": sdf.d2.rectangle(size[[0, 1]]),
            "area": size[0] * size[1],
            "bb": size[[0, 1]],
            "height": size[2],
        }
        inner_params = {
            "containment": True,
            "shape": sdf.d2.rectangle(size[[0, 1]] - 2 * WALL_THICKNESS),
            "area":
            (size[0] - 2 * WALL_THICKNESS) * (size[1] - 2 * WALL_THICKNESS),
            "bb": size[[0, 1]] - 2 * WALL_THICKNESS,
            "height": size[2] - 2 * WALL_THICKNESS,
        }
        aff_dicts = [{
            "name":
            "drawer_outer_bottom",
            "tf":
            self.transform @ np.array((
                (1, 0, 0, 0),
                (0, -1, 0, 0),
                (0, 0, -1, -self.size[2] / 2.0),
                (0, 0, 0, 1),
            )),
            "params":
            outer_params
        }, {
            "name":
            "drawer_inner_bottom",
            "tf":
            self.transform @ np.array((
                (1, 0, 0, WALL_THICKNESS),
                (0, 1, 0, 0),
                (0, 0, 1, -self.size[2] / 2.0 + 2 * WALL_THICKNESS),
                (0, 0, 0, 1),
            )),
            "params":
            inner_params,
        }]
        for aff in aff_dicts:
            self.add_aff(
                Affordance(name=aff["name"],
                           transform=aff["tf"],
                           **aff["params"]))

    def create_bullet_shapes(self, global_transform):
        visual_shapes = []
        collision_shapes = []
        halfwlExtents = [
            self.size[0] / 2., self.size[1] / 2., WALL_THICKNESS / 4.
        ]
        halflhExtents = [
            self.size[0] / 2., WALL_THICKNESS / 4., self.size[2] / 2.
        ]
        halfwhExtents = [
            WALL_THICKNESS / 4., self.size[1] / 2., self.size[2] / 2.
        ]
        translation, quaternion = self.parse_transform(global_transform)

        shape_params = [
            {
                'ext': halfwlExtents,
                'frame_position':
                [0, 0, self.size[2] / 2. + WALL_THICKNESS / 4.]
            },
            {
                'ext': halfwlExtents
            },
            {
                'ext': halflhExtents
            },
            {
                'ext': halflhExtents
            },
            {
                'ext': halfwhExtents
            },
            {
                'ext': halfwhExtents
            },
        ]

        for param in shape_params:
            visual_shapes.append(
                p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=param["ext"],
                    visualFramePosition=param.get("frame_position", None),
                ))
            collision_shapes.append(
                p.createCollisionShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=param["ext"],
                    # halfExtents=[0, 0, 0],
                    collisionFramePosition=param.get("frame_position", None),
                ))

        visual_shapes.append(
            p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.015))
        collision_shapes.append(
            p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=0.001))
        multibody = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shapes[0],
            baseVisualShapeIndex=visual_shapes[0],
            basePosition=translation + BULLET_GROUND_OFFSET,
            baseOrientation=np.hstack((quaternion[1:4], quaternion[0])),
            linkMasses=[0, 1, 0, 0, 0, 0],
            linkCollisionShapeIndices=collision_shapes[1:],
            linkVisualShapeIndices=visual_shapes[1:],
            linkPositions=[
                # [0, 0, self.size[2] / 2. - WALL_THICKNESS / 4.],
                [
                    0, -self.size[1] / 2. + WALL_THICKNESS / 4.,
                    -self.size[2] / 2. + WALL_THICKNESS / 4.
                ],
                [0, self.size[1] / 2. - WALL_THICKNESS / 4., 0],
                [0, -self.size[1] + WALL_THICKNESS / 2., 0],
                [
                    -self.size[0] / 2. + WALL_THICKNESS / 4.,
                    -self.size[1] / 2. + WALL_THICKNESS / 4., 0
                ],
                [
                    self.size[0] / 2. - WALL_THICKNESS / 4.,
                    -self.size[1] / 2. + WALL_THICKNESS / 4., 0
                ],
                [0, 0.01, 0],
            ],
            linkOrientations=[
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ],
            linkInertialFramePositions=[
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            linkInertialFrameOrientations=[
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ],
            linkParentIndices=[2, 0, 2, 2, 2, 2],
            linkJointTypes=[
                p.JOINT_FIXED,
                p.JOINT_PRISMATIC,
                p.JOINT_FIXED,
                p.JOINT_FIXED,
                p.JOINT_FIXED,
                p.JOINT_PRISMATIC,
            ],
            linkJointAxis=[
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            useMaximalCoordinates=False,
        )
        p.changeDynamics(multibody, 1, jointLowerLimit=100, jointUpperLimit=0)
        p.changeVisualShape(multibody,
                            linkIndex=4,
                            rgbaColor=[0.76470588, 0.765, 0.765, 1.],
                            specularColor=[0.4, 0.4, 0])
        return visual_shapes, collision_shapes, multibody