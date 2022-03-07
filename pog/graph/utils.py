import numpy as np
import transforms3d as tf3d
import math
import logging

from pog.graph.params import OFFSET

__all__ = [ 'affTFxy',
            'affTFxy2dof',
            'p2cTF']

def affTFxy(pose):
    """Compute parent to child affordance transformation given pose between affordances

    Args:
        pose (list): x-y-theta pose from parent to child

    Returns:
        4x4 homogeneous matrix: Parent to child affordance transformation
    """
    try:
        x, y, theta = pose[0], pose[1], pose[2]
    except:
        try:
            affTFxy2dof(pose)
        except:
            logging.error('Pose x-y-theta is required.')
    Rmat = tf3d.axangles.axangle2mat([0,0,1], theta)
    Rmat = tf3d.axangles.axangle2mat([0,1,0], math.pi) @ Rmat

    Tvec = [x,y,OFFSET]

    return tf3d.affines.compose(Tvec, Rmat, np.ones(3))

def affTFxy2dof(pose):
    """Compute parent to child affordance transformation given pose between affordances

    Args:
        pose (list): x-y pose from parent to child

    Returns:
        4x4 homogeneous matrix: Parent to child affordance transformation
    """
    try:
        x, y = pose[0], pose[1]
    except:
        logging.error('Pose x-y is required.')
    
    Rmat = tf3d.axangles.axangle2mat([0,1,0], math.pi)
    Tvec = [x,y,OFFSET]

    return tf3d.affines.compose(Tvec, Rmat, np.ones(3))

def p2cTF(parent_afftf, child_afftf, afftf):
    """Compute parent to child transformation

    Args:
        parent_afftf (4x4 numpy array): transformation from parent object frame to parent affordance frame
        child_afftf (4x4 numpy array): transformation from child object frame to child affordance frame
        afftf (4x4 numpy array): transformation from parent affordance to child affordance frame

    Returns:
        4x4 numpy array: parent to child transformation
    """
    return np.dot(np.dot(parent_afftf, afftf), np.linalg.inv(child_afftf))

def match(obj1, obj2):
    return obj1 == obj2