import numpy as np
import os
import shutil
from PIL import Image
import cv2


original_data_root = '/local-scratch/shitaot/generalized_nerf/data/scannet/'
original_data_dir = os.path.join(original_data_root, 'train')
output_data_dir = './processed_train'

def filter_valid_id(data_dir, scene_name, id_list):
    empty_lst=[]
    filter_path=os.path.join(os.path.dirname(data_dir), 'filter', scene_name+'.txt')
    filter_id_list=set()
    if os.path.exists(filter_path):
        with open(filter_path) as f:
            for line in f.readlines():
                filter_id_list.add(int(line.strip('\n')))
    else:
        raise ValueError('Do not find the filter file at {}'.format(filter_path))
    for id in id_list:
        if id not in filter_id_list:
            empty_lst.append(id)
  
    return empty_lst

def batched_angular_dist_rot_matrix(R1, R2):
    '''
    calculate the angular distance between two rotation matrices (batched)
    :param R1: the first rotation matrix [N, 3, 3]
    :param R2: the second rotation matrix [N, 3, 3]
    :return: angular distance in radiance [N, ]
    '''
    TINY_NUMBER = 1e-6
    assert R1.shape[-1] == 3 and R2.shape[-1] == 3 and R1.shape[-2] == 3 and R2.shape[-2] == 3
    return np.arccos(np.clip((np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1) / 2.,
                             a_min=-1 + TINY_NUMBER, a_max=1 - TINY_NUMBER))

def get_nearest_pose_ids(tar_pose, ref_poses, num_select, tar_id=-1, angular_dist_method='vector',
                         scene_center=(0, 0, 0)):
    '''
    Args:
        tar_pose: target pose [3, 3]
        ref_poses: reference poses [N, 3, 3]
        num_select: the number of nearest views to select
    Returns: the selected indices
    '''
    num_cams = len(ref_poses)
    num_select = min(num_select, num_cams-1)
    batched_tar_pose = tar_pose[None, ...].repeat(num_cams, 0)

    if angular_dist_method == 'matrix':
        dists = batched_angular_dist_rot_matrix(batched_tar_pose[:, :3, :3], ref_poses[:, :3, :3])
    elif angular_dist_method == 'vector':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        scene_center = np.array(scene_center)[None, ...]
        tar_vectors = tar_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
    elif angular_dist_method == 'dist':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)
    elif angular_dist_method == 'dist+matrix':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        loc_dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)
        angle_dists = batched_angular_dist_rot_matrix(batched_tar_pose[:, :3, :3], ref_poses[:, :3, :3])
        dists = loc_dists * 0.8 + angle_dists * 0.2
    else:
        raise Exception('unknown angular distance calculation method!')

    if tar_id >= 0:
        assert tar_id < num_cams
        dists[tar_id] = 1e3  # make sure not to select the target id itself

    return dists
    sorted_ids = np.argsort(dists)
    selected_ids = sorted_ids[:num_select]
    # print(angular_dists[selected_ids] * 180 / np.pi)
    return selected_ids


for scene_name in os.listdir(original_data_dir):
    print(scene_name)
    scene_path = os.path.join(original_data_dir, scene_name)
    if os.path.isdir(scene_path) is False:
        continue
    rgb_dir = os.path.join(scene_path, 'color')
    depth_dir = os.path.join(scene_path, 'depth')
    pose_dir = os.path.join(scene_path, 'pose') # extrinsic

    output_scene_path = os.path.join(output_data_dir, scene_name)
    os.makedirs(output_scene_path, exist_ok=True)

    image_ids = [t for t in range(len(os.listdir(rgb_dir)))]
    all_id_lists = filter_valid_id(original_data_root, scene_name, image_ids)

    all_id_lists = all_id_lists[::5]

    all_poses = []
    for idx, id_ in enumerate(all_id_lists):
        # image
        src_dir = os.path.join(rgb_dir, f'{id_}.jpg') 
        # resize
        im = cv2.imread(src_dir)
        original_shape = im.shape[:2]
        im = cv2.resize(im, (640, 480))
        dst_dir = os.path.join(output_scene_path, f'im_{idx:08}.jpg')
        cv2.imwrite(dst_dir, im)
        #shutil.copy(src_dir,dst_dir)

        # dm.npy
        im = Image.open(os.path.join(depth_dir, f'{id_}.png'))
        dm = np.array(im) / 1000.
        np.save(os.path.join(output_scene_path, f'dm_{idx:08}'), dm)


        f = open(os.path.join(pose_dir, f'{id_}.txt'), 'r')
        pose = []
        for line in f.readlines():
            row = [float(v.strip()) for v in line.split(' ')]
            pose.append(row)
        pose = np.array(pose)
        all_poses.append(pose)
        f.close()

        # count new
        if 'test' in original_data_dir:
            skip = True
            for tt in range(id_-3, id_+4):
                if os.path.exists(os.path.join(scene_path, 'count', f'cnt_{tt}.npy')):
                    skip = False
            if skip:
                count = np.zeros(len(all_id_lists))
                np.save(os.path.join(output_scene_path, f'count_{idx:08}'), count)
                continue
        if os.path.exists(os.path.join(scene_path, 'count', f'cnt_{id_}.npy')):
            dd = np.load(os.path.join(scene_path, 'count', f'cnt_{id_}.npy'), allow_pickle=True).tolist()
        else:
            for deta_id in [-1,+1,-2,+2,-3,+3,-4,+4,-5,+5]:
                temp_id = deta_id + id_
                if os.path.exists(os.path.join(scene_path, 'count', f'cnt_{temp_id}.npy')):
                    dd = np.load(os.path.join(scene_path, 'count', f'cnt_{temp_id}.npy'), allow_pickle=True).tolist()
                    break
        
        count = []
        for other_idx in range(len(all_id_lists)):
            if other_idx == idx:
                count.append(-1)
            else:
                other_id = all_id_lists[other_idx]
                if 'test' in original_data_dir and other_id > id_:
                    overlap = 0
                else:
                    temp_idx = np.argmin(np.abs(dd['idx'] - other_id))
                    overlap = dd['overlap_cnt'][temp_idx]
                count.append(overlap)
        count = np.array(count)
        np.save(os.path.join(output_scene_path, f'count_{idx:08}'), count)

    
    all_poses = np.array(all_poses)
    # Rs
    rs = all_poses[:,:3,:3]
    np.save(os.path.join(output_scene_path, 'Rs.npy'), rs)
    # ts
    ts = all_poses[:,:3, 3]
    np.save(os.path.join(output_scene_path, 'ts.npy'), ts)

    #    ori_img_shape = [3, 968, 1296]
    #    intrinsic[0, :] *= (self.width / ori_img_shape[2])
    #    intrinsic[1, :] *= (self.height / ori_img_shape[1])

    # Ks.npy
    f = open(os.path.join(scene_path, 'intrinsic', 'intrinsic_depth.txt'), 'r')
    k = []
    for line in f.readlines()[:-1]:
        row = [float(v) for v in line.split(' ')[:-1]]
        k.append(row)
    f.close()
    k = np.array(k)
    # scale
    k[0,:] *= 640 / original_shape[1]
    k[1,:] *= 480 / original_shape[0]
    k = np.stack([k]*len(all_id_lists))
    np.save(os.path.join(output_scene_path, 'Ks.npy'), k)

    # count
    #for idx in range(all_poses.shape[0]):
    #    count_ = get_nearest_pose_ids(all_poses[idx, :3, :3], 
    #                         all_poses[:,:3,:3], 10, 
    #                         angular_dist_method='matrix', tar_id=idx)
    #    np.save(os.path.join(output_scene_path, f'count_{idx:08}'), count_)

    
    

