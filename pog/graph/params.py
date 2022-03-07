# Parameter file for pog.graph
OFFSET = 0.003  # Offset (z-axis) between two objects (avoids collision between parent and child)
MAX_STABLE_POSES = 3  # Maximum stable poses to be stored for an arbitrary object.
# BULLET_GROUND_OFFSET = [0., 0., -0.0]

# for pog.graph.chromosome
FRICTION_ANGLE_THRESH = 0.1
MAX_INITIAL_TRIES = 100
TF_DIFF_THRESH = 0.01

# {support up: support down}
PairedSurface = {
    "box_aff_pz": "box_aff_nz",
    "box_aff_nz": "box_aff_pz",
    "box_aff_px": "box_aff_nx",
    "box_aff_nx": "box_aff_px",
    "box_aff_py": "box_aff_ny",
    "box_aff_ny": "box_aff_py",
    "cylinder_aff_nz": "cylinder_aff_pz",
    "cylinder_aff_pz": "cylinder_aff_nz",
    "shelf_aff_pz_top": "shelf_aff_nz",
    "shelf_aff_pz_bottom": "shelf_aff_nz",
    "shelf_outer_top": "shelf_outer_bottom",
    "cabinet_inner_bottom" : "cabinet_outer_bottom",
    "cabinet_inner_middle" : "cabinet_outer_bottom",
    "drawer_inner_bottom" : "drawer_outer_bottom",
}

ContainmentSurface = [
    'shelf_aff_pz_bottom', 'cabinet_inner_bottom', 'cabinet_inner_middle',
    'drawer_inner_bottom'
]

WALL_THICKNESS = 0.02

COLOR_SOLID = {
    "dark grey": [0.34509804, 0.34509804, 0.34509804, 1.],
    "light grey": [0.76470588, 0.76470588, 0.76470588, 1.],
    "red": [0.9254902, 0.10980392, 0.14117647, 1.],
    "blue": [0.24705882, 0.28235294, 0.8, 0.8],
    "green": [0.05490196, 0.81960784, 0.27058824, 1.],
    "yellow": [1., 0.94901961, 0., 1.],
    "purple": [0.72156863, 0.23921569, 0.72941176, 1.],
    "brown": [0.7255, 0.4784, 0.3373, 1.],
    "transparent": [0, 0, 0, 0],
    "red-trans": [0.9254902, 0.10980392, 0.14117647, 1.0],
    "blue-trans": [0.24705882, 0.28235294, 0.8, 1.0],
}

COLOR_IMAGE = {
    "dark grey": [0.34509804, 0.34509804, 0.34509804, 1.],
    "light grey": [0.76470588, 0.76470588, 0.76470588, 1.],
    "red": [0.9254902, 0.10980392, 0.14117647, 1.],
    "blue": [0.24705882, 0.28235294, 0.8, 0.8],
    "green": [0.05490196, 0.81960784, 0.27058824, 1.],
    "yellow": [1., 0.94901961, 0., 1.],
    "purple": [0.72156863, 0.23921569, 0.72941176, 1.],
    "brown": [0.7255, 0.4784, 0.3373, 1.],
    "transparent": [0, 0, 0, 0],
    "red-trans": [0.9254902, 0.10980392, 0.14117647, 1.0],
    "blue-trans": [0.24705882, 0.28235294, 0.8, 1.0],
}

COLOR_EXP = {
    "light grey": [0.76470588, 0.76470588, 0.76470588, 1.],
    "red": [0.9254902, 0.10980392, 0.14117647, 1.],
    "blue": [0., 0.4470, 0.7410, 1.],
    "orange" : [0.8500, 0.3250, 0.0980, 1.],
    "green": [0.4660, 0.6740, 0.1880, 1.],
    "yellow": [0.9290, 0.6940, 0.1250, 1.],
    "purple": [0.4940, 0.1840, 0.5560, 1.],
    "brown": [0.7255, 0.4784, 0.3373, 1.],
    "transparent": [0, 0, 0, 0],
    "light blue": [0.3010, 0.7450, 0.9330, 1.0],
    "dark red": [0.6350, 0.0780, 0.1840, 1.],
    "dark grey": [0.34509804, 0.34509804, 0.34509804, 1.],
}

COLOR = COLOR_EXP

COLOR_DICT = {
    0: "transparent",
    1: "light grey",
    2: "dark grey",
    3: "yellow",
    4: "orange",
    5: "dark red",
    6: "purple",
    7: "green",
    8: "brown",
    9: "blue",
    99: 'brown'
}

BULLET_GROUND_OFFSET = [0.5, -0.5, 0.]
