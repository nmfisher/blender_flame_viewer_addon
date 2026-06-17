"""Pure-numpy FLAME forward pass. No Blender or PyTorch dependencies."""

import numpy as np
from scipy import sparse as sp


def batch_rodrigues(rot_vecs):
    """Axis-angle to rotation matrices. (N,3) -> (N,3,3)."""
    dtype = rot_vecs.dtype
    batch = rot_vecs.shape[0]
    device = rot_vecs.device if hasattr(rot_vecs, 'device') else None

    angle = np.linalg.norm(rot_vecs + 1e-8, axis=1, keepdims=True)
    rot_dir = rot_vecs / angle

    cos = np.cos(angle)[:, None]
    sin = np.sin(angle)[:, None]

    rx, ry, rz = rot_dir[:, 0:1], rot_dir[:, 1:2], rot_dir[:, 2:3]

    K = np.zeros((batch, 3, 3), dtype=dtype)
    K[:, 0, 1] = -rz[:, 0]
    K[:, 0, 2] = ry[:, 0]
    K[:, 1, 0] = rz[:, 0]
    K[:, 1, 2] = -rx[:, 0]
    K[:, 2, 0] = -ry[:, 0]
    K[:, 2, 1] = rx[:, 0]

    ident = np.eye(3, dtype=dtype)[None]
    R = ident + sin * K + (1 - cos) * np.einsum('nij,njk->nik', K, K)
    return R


def _transform_mat(R, t):
    """Build 4x4 homogeneous transform from rotation (N,3,3) and translation (N,3)."""
    N = R.shape[0]
    T = np.zeros((N, 4, 4), dtype=R.dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1.0
    return T


def batch_rigid_transform(rot_mats, joints, parents):
    """Kinematic tree traversal. (J,3,3), (J,3), (J,) -> (J,4,4), (J,3)."""
    J = joints.shape[0]

    rel_joints = joints.copy()
    rel_joints[1:] -= joints[parents[1:]]

    local_transforms = _transform_mat(rot_mats, rel_joints)

    transforms = np.zeros((J, 4, 4), dtype=rot_mats.dtype)
    transforms[0] = local_transforms[0]
    for i in range(1, J):
        transforms[i] = transforms[parents[i]] @ local_transforms[i]

    posed_joints = transforms[:, :3, 3]

    # Subtract rest-pose joint offset from translation column.
    # smplx pads joints (B,J,3,1) to (B,J,4,1) with a 0 (not a 1),
    # so the 4th row of the homogeneous coord is 0, preserving the
    # (3,3) element of the 4x4 transforms.
    joints_homo = np.concatenate(
        [joints, np.zeros((J, 1), dtype=joints.dtype)], axis=1
    )  # (J, 4) — 4th component is 0, matching smplx F.pad behavior
    offset = np.einsum('jik,jk->ji', transforms, joints_homo)  # (J, 4)
    offset_mat = np.concatenate(
        [np.zeros((J, 4, 3), dtype=joints.dtype), offset[:, :, None]], axis=2
    )  # (J, 4, 4)
    rel_transforms = transforms - offset_mat

    return rel_transforms, posed_joints


def flame_forward(shape_params, expr_params, pose_params, neck_pose,
                  transl, model, eye_pose=None):
    """Reconstruct FLAME vertices from parameters (numpy, batch=1).

    Parameters
    ----------
    shape_params : (300,) or less — identity PCA coefficients
    expr_params  : (100,) or less — expression PCA coefficients
    pose_params  : (6,) — global rotation (3) + jaw rotation (3)
    neck_pose    : (3,) — neck rotation
    transl       : (3,) — translation
    model        : dict with keys: v_template, shapedirs, posedirs_r,
                   J_regressor, parents, weights
    eye_pose     : (6,) or None — left eye (3) + right eye (3); None → zeros

    Returns
    -------
    verts : (V, 3) float64
    """
    v_template = model['v_template']
    shapedirs = model['shapedirs']
    posedirs_r = model['posedirs_r']  # already (36, V*3)
    J_regressor = model['J_regressor']
    parents = model['parents']
    lbs_weights = model['weights']
    n_verts = v_template.shape[0]

    # Pad betas to 400
    betas = np.zeros(400, dtype=np.float64)
    betas[:len(shape_params)] = shape_params
    betas[300:300 + len(expr_params)] = expr_params

    # Assemble full_pose: 5 joints x 3 = 15
    # [global(3), neck(3), jaw(3), eye_L(3), eye_R(3)]
    if eye_pose is None:
        eye_pose = np.zeros(6, dtype=np.float64)
    full_pose = np.concatenate([
        pose_params[:3],
        neck_pose,
        pose_params[3:6],
        eye_pose,
    ])

    # 1. Shape blend
    v_shaped = v_template + np.einsum('l,mkl->mk', betas, shapedirs)

    # 2. Joints
    if sp.issparse(J_regressor):
        joints = J_regressor.dot(v_shaped)
    else:
        joints = J_regressor @ v_shaped

    # 3a. Rodrigues
    rot_mats = batch_rodrigues(full_pose.reshape(-1, 3)).reshape(-1, 3, 3)

    # 3b. Pose blend
    pose_feature = (rot_mats[1:] - np.eye(3, dtype=np.float64)).reshape(1, -1)
    pose_offsets = (pose_feature @ posedirs_r).reshape(n_verts, 3)
    v_posed = v_shaped + pose_offsets

    # 4. Rigid transforms
    A, _ = batch_rigid_transform(rot_mats, joints, parents)

    # 5. LBS
    W = lbs_weights  # (V, J)
    T = (W @ A.reshape(A.shape[0], 16)).reshape(n_verts, 4, 4)
    v_posed_homo = np.concatenate([v_posed, np.ones((n_verts, 1))], axis=1)
    v_homo = np.einsum('vij,vj->vi', T, v_posed_homo)
    verts = v_homo[:, :3]

    # 6. Translation
    verts += transl[None, :]

    return verts


def load_flame_model(pkl_path):
    """Load a FLAME pickle and return a model dict ready for flame_forward().

    Converts posedirs from (V,3,36) to (36,V*3) at load time.
    """
    import pickle
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f, encoding='latin1')

    J_regressor = raw['J_regressor']
    if sp.issparse(J_regressor):
        pass  # keep sparse
    else:
        J_regressor = np.asarray(J_regressor, dtype=np.float64)

    num_pose_basis = raw['posedirs'].shape[-1]
    posedirs_r = np.reshape(raw['posedirs'], [-1, num_pose_basis]).T.astype(np.float64)

    parents = np.asarray(raw['kintree_table'][0]).astype(np.int64)
    parents[0] = -1

    model = {
        'v_template': np.asarray(raw['v_template'], dtype=np.float64),
        'shapedirs': np.asarray(raw['shapedirs'], dtype=np.float64),
        'posedirs_r': posedirs_r,
        'J_regressor': J_regressor,
        'parents': parents,
        'weights': np.asarray(raw['weights'], dtype=np.float64),
        'f': np.asarray(raw['f'], dtype=np.int32),
    }
    return model
