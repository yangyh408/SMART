import os
import torch
import pickle
import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

from smart.model import SMART
from smart.utils.config import load_config_act
from smart.utils.log import Logging

SMART_DIR = "/home/yangyh408/codes/SMART"

config = load_config_act(os.path.join(SMART_DIR, "configs/validation/validation_scalable.yaml"))
pretrain_ckpt = os.path.join(SMART_DIR, "ckpt/20241021_1037/epoch=07-step=30440-val_loss=2.52.ckpt")
Predictor = SMART
logger = Logging().log(level='DEBUG')
model = Predictor(config.Model)
model.load_params_from_file(filename=pretrain_ckpt, logger=logger)

current_step = 10
shift = 5
noise = True
training = False
argmin_sample_len = 3

map_token_traj_path = os.path.join(SMART_DIR, "smart/tokens/map_traj_token5.pkl")
map_token_traj = pickle.load(open(map_token_traj_path, 'rb'))
map_token = {'traj_src': map_token_traj['traj_src'], }
traj_end_theta = np.arctan2(map_token['traj_src'][:, -1, 1]-map_token['traj_src'][:, -2, 1],
                            map_token['traj_src'][:, -1, 0]-map_token['traj_src'][:, -2, 0])
# 生成从 start 到 end 的 steps 个等间隔值。
indices = torch.linspace(0, map_token['traj_src'].shape[1]-1, steps=argmin_sample_len).long()
map_token['sample_pt'] = torch.from_numpy(map_token['traj_src'][:, indices]).to(torch.float)
map_token['traj_end_theta'] = torch.from_numpy(traj_end_theta).to(torch.float)
map_token['traj_src'] = torch.from_numpy(map_token['traj_src']).to(torch.float)

agent_token_path = os.path.join(SMART_DIR, "smart/tokens/cluster_frame_5_2048.pkl")
agent_token_data = pickle.load(open(agent_token_path, 'rb'))
trajectory_token = agent_token_data['token']
trajectory_token_traj = agent_token_data['traj']
trajectory_token_all = agent_token_data['token_all']
# 对所有token依据倒数第二帧的状态为基准状态对最后一帧进行归一化
token_last_all = {}
for k, v in trajectory_token_all.items():
    # 计算每个 agent 的最终 token 朝向
    token_last = torch.from_numpy(v[:, -2:]).to(torch.float)    # [2048, 2, 4, 2]
    diff_xy = token_last[:, 0, 0] - token_last[:, 0, 3]         # 倒数第二帧 左前-左后
    theta = torch.arctan2(diff_xy[:, 1], diff_xy[:, 0])         # 倒数第二帧的航向角
    cos, sin = theta.cos(), theta.sin()
    # 生成旋转矩阵
    rot_mat = theta.new_zeros(token_last.shape[0], 2, 2)
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = -sin
    rot_mat[:, 1, 0] = sin
    rot_mat[:, 1, 1] = cos
    # 应用旋转矩阵并归一化 token 数据
    agent_token = torch.bmm(token_last[:, 1], rot_mat)
    agent_token -= token_last[:, 0].mean(1)[:, None, :]
    token_last_all[k] = agent_token.numpy()

def clean_heading(data):
    """
        这个函数 clean_heading 的主要功能是对“heading” (朝向角度) 进行清理，以修复明显异常或突然变化的朝向角度
        （例如，当相邻帧之间的朝向差异超过一定阈值时），从而平滑朝向数据。
        具体而言，代码通过对相邻帧的朝向差异进行检测和修正，使得朝向变化更连贯。
    """
    heading = data['agent']['heading']
    valid = data['agent']['valid_mask']
    pi = torch.tensor(torch.pi)
    n_vehicles, n_frames = heading.shape

    heading_diff_raw = heading[:, :-1] - heading[:, 1:]
    heading_diff = torch.remainder(heading_diff_raw + pi, 2 * pi) - pi
    heading_diff[heading_diff > pi] -= 2 * pi
    heading_diff[heading_diff < -pi] += 2 * pi

    valid_pairs = valid[:, :-1] & valid[:, 1:]

    for i in range(n_frames - 1):
        change_needed = (torch.abs(heading_diff[:, i:i + 1]) > 1.0) & valid_pairs[:, i:i + 1]

        heading[:, i + 1][change_needed.squeeze()] = heading[:, i][change_needed.squeeze()]

        if i < n_frames - 2:
            heading_diff_raw = heading[:, i + 1] - heading[:, i + 2]
            heading_diff[:, i + 1] = torch.remainder(heading_diff_raw + pi, 2 * pi) - pi
            heading_diff[heading_diff[:, i + 1] > pi] -= 2 * pi
            heading_diff[heading_diff[:, i + 1] < -pi] += 2 * pi

def cal_polygon_contour(x, y, theta, width, length):
    """
        函数功能：计算一个矩形多边形的四个顶点坐标（轮廓）
        返回值：返回一个形状为 [n, 4, 2] 的数组 polygon_contour，表示每个矩形的四个顶点的坐标，方便后续用作绘制或碰撞检测等应用。
    """
    left_front_x = x + 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
    left_front_y = y + 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)
    left_front = np.column_stack((left_front_x, left_front_y))

    right_front_x = x + 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
    right_front_y = y + 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
    right_front = np.column_stack((right_front_x, right_front_y))

    right_back_x = x - 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
    right_back_y = y - 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
    right_back = np.column_stack((right_back_x, right_back_y))

    left_back_x = x - 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
    left_back_y = y - 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)
    left_back = np.column_stack((left_back_x, left_back_y))

    polygon_contour = np.concatenate(
        (left_front[:, None, :], right_front[:, None, :], right_back[:, None, :], left_back[:, None, :]), axis=1)

    return polygon_contour

def match_token(pos, valid_mask, heading, category, agent_category, extra_mask):
    """
        将轨迹位置和朝向数据与预定义的 token 数据进行匹配，以便在场景中的每个时间步中都能追踪到正确的 token。
    """
    agent_token_src = trajectory_token[category]
    token_last = token_last_all[category]
    if shift <= 2:
        if category == 'veh':
            width = 1.0
            length = 2.4
        elif category == 'cyc':
            width = 0.5
            length = 1.5
        else:
            width = 0.5
            length = 0.5
    else:
        if category == 'veh':
            width = 2.0
            length = 4.8
        elif category == 'cyc':
            width = 1.0
            length = 2.0
        else:
            width = 1.0
            length = 1.0

    prev_heading = heading[:, 0]
    prev_pos = pos[:, 0]
    agent_num, num_step, feat_dim = pos.shape   # [NA, 91, 2]
    token_num, token_contour_dim, feat_dim = agent_token_src.shape  # [2048, 4, 2]
    agent_token_src = agent_token_src.reshape(1, token_num * token_contour_dim, feat_dim).repeat(agent_num, 0)
    token_last = token_last.reshape(1, token_num * token_contour_dim, feat_dim).repeat(extra_mask.sum(), 0)
    token_index_list = []
    token_contour_list = []
    prev_token_idx = None

    for i in range(shift, pos.shape[1], shift):
        # 上一token所在位置航向角（5帧前）
        theta = prev_heading
        # 当前航向角和位置
        cur_heading = heading[:, i]
        cur_pos = pos[:, i]
        # 将归一化的原始token信息以上一时刻位置和航向状态为基准调整到全局坐标系
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(agent_num, 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        agent_token_world = torch.bmm(torch.from_numpy(agent_token_src).to(torch.float), rot_mat).reshape(agent_num,
                                                                                                            token_num,
                                                                                                            token_contour_dim,
                                                                                                            feat_dim)
        agent_token_world += prev_pos[:, None, None, :]

        # 获取当前所在位置的矩形四角信息
        cur_contour = cal_polygon_contour(cur_pos[:, 0], cur_pos[:, 1], cur_heading, width, length)
        # 找出与当前距离最近的token作为匹配对象，记录该tokenid
        agent_token_index = torch.from_numpy(np.argmin(
            np.mean(np.sqrt(np.sum((cur_contour[:, None, ...] - agent_token_world.numpy()) ** 2, axis=-1)), axis=2),
            axis=-1))
        if prev_token_idx is not None and noise:
            same_idx = prev_token_idx == agent_token_index
            same_idx[:] = True
            topk_indices = np.argsort(
                np.mean(np.sqrt(np.sum((cur_contour[:, None, ...] - agent_token_world.numpy()) ** 2, axis=-1)),
                        axis=2), axis=-1)[:, :5]
            sample_topk = np.random.choice(range(0, topk_indices.shape[1]), topk_indices.shape[0])
            agent_token_index[same_idx] = \
                torch.from_numpy(topk_indices[np.arange(topk_indices.shape[0]), sample_topk])[same_idx]
        # 将匹配的tokenid转换为矩形四角坐标
        token_contour_select = agent_token_world[torch.arange(agent_num), agent_token_index]

        # 将当前帧信息更新为上一帧信息
        diff_xy = token_contour_select[:, 0, :] - token_contour_select[:, 3, :]
        # 数据集中原航向角
        prev_heading = heading[:, i].clone()
        # 如果是这一帧被预测的对象，则用当前token所在状态更新航向和位置信息
        prev_heading[valid_mask[:, i - shift]] = torch.arctan2(diff_xy[:, 1], diff_xy[:, 0])[
            valid_mask[:, i - shift]]

        prev_pos = pos[:, i].clone()
        prev_pos[valid_mask[:, i - shift]] = token_contour_select.mean(dim=1)[valid_mask[:, i - shift]]
        prev_token_idx = agent_token_index
        token_index_list.append(agent_token_index[:, None])
        token_contour_list.append(token_contour_select[:, None, ...])

    token_index = torch.cat(token_index_list, dim=1)
    token_contour = torch.cat(token_contour_list, dim=1)

    # extra matching（如果在第十一帧存在但第六帧不存在的代理，则根据第十帧的状态来匹配token信息）
    if not training:
        theta = heading[extra_mask, current_step - 1]
        prev_pos = pos[extra_mask, current_step - 1]
        cur_pos = pos[extra_mask, current_step]
        cur_heading = heading[extra_mask, current_step]
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(extra_mask.sum(), 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        agent_token_world = torch.bmm(torch.from_numpy(token_last).to(torch.float), rot_mat).reshape(
            extra_mask.sum(), token_num, token_contour_dim, feat_dim)
        agent_token_world += prev_pos[:, None, None, :]

        cur_contour = cal_polygon_contour(cur_pos[:, 0], cur_pos[:, 1], cur_heading, width, length)
        agent_token_index = torch.from_numpy(np.argmin(
            np.mean(np.sqrt(np.sum((cur_contour[:, None, ...] - agent_token_world.numpy()) ** 2, axis=-1)), axis=2),
            axis=-1))
        token_contour_select = agent_token_world[torch.arange(extra_mask.sum()), agent_token_index]

        token_index[extra_mask, 1] = agent_token_index
        token_contour[extra_mask, 1] = token_contour_select

    return token_index, token_contour

def tokenize_agent(data):
    if data['agent']["velocity"].shape[1] == 90:
        print(data['scenario_id'], data['agent']["velocity"].shape)
    
    # 创建插值掩码 interplote_mask，用于标记那些当前时间步为无效但坐标非零的位置，以确定需要插值的数据点
    interplote_mask = (data['agent']['valid_mask'][:, current_step] == False) * (
            data['agent']['position'][:, current_step, 0] != 0)
    # 通过检查当前时间步中无效但位置非零的轨迹点，将其前一个时间步的位置、速度、航向等信息进行估算和填充，确保轨迹数据连续性
    if data['agent']["velocity"].shape[-1] == 2:
        data['agent']["velocity"] = torch.cat([data['agent']["velocity"],
                                                torch.zeros(data['agent']["velocity"].shape[0],
                                                            data['agent']["velocity"].shape[1], 1)], dim=-1)
    vel = data['agent']["velocity"][interplote_mask, current_step]
    # 插值前一个时间步的位置、航向、速度
    data['agent']['position'][interplote_mask, current_step - 1, :3] = data['agent']['position'][
                                                                            interplote_mask, current_step,
                                                                            :3] - vel * 0.1
    data['agent']['heading'][interplote_mask, current_step - 1] = data['agent']['heading'][
        interplote_mask, current_step]
    data['agent']["velocity"][interplote_mask, current_step - 1] = data['agent']["velocity"][
        interplote_mask, current_step]
    data['agent']['valid_mask'][interplote_mask, current_step - 1:current_step + 1] = True

    data['agent']['type'] = data['agent']['type'].to(torch.uint8)

    clean_heading(data)
    matching_extra_mask = (data['agent']['valid_mask'][:, current_step] == True) * (
            data['agent']['valid_mask'][:, current_step - 5] == False)

    interplote_mask_first = (data['agent']['valid_mask'][:, 0] == False) * (data['agent']['position'][:, 0, 0] != 0)
    data['agent']['valid_mask'][interplote_mask_first, 0] = True

    agent_pos = data['agent']['position'][:, :, :2]
    valid_mask = data['agent']['valid_mask']
    # 以下标1为起点，长度为6，间隔为5创建滑动窗口
    valid_mask_shift = valid_mask.unfold(1, shift + 1, shift)         # [NA, 18, 6]
    # 每个滑动窗口的起止都为true时窗口才有效
    token_valid_mask = valid_mask_shift[:, :, 0] * valid_mask_shift[:, :, -1]   # [NA, 18]
    agent_type = data['agent']['type']
    agent_category = data['agent']['category']
    agent_heading = data['agent']['heading']
    vehicle_mask = agent_type == 0
    cyclist_mask = agent_type == 2
    ped_mask = agent_type == 1

    veh_pos = agent_pos[vehicle_mask, :, :]
    veh_valid_mask = valid_mask[vehicle_mask, :]
    cyc_pos = agent_pos[cyclist_mask, :, :]
    cyc_valid_mask = valid_mask[cyclist_mask, :]
    ped_pos = agent_pos[ped_mask, :, :]
    ped_valid_mask = valid_mask[ped_mask, :]

    veh_token_index, veh_token_contour = match_token(veh_pos, veh_valid_mask, agent_heading[vehicle_mask],
                                                            'veh', agent_category[vehicle_mask],
                                                            matching_extra_mask[vehicle_mask])
    ped_token_index, ped_token_contour = match_token(ped_pos, ped_valid_mask, agent_heading[ped_mask], 'ped',
                                                            agent_category[ped_mask], matching_extra_mask[ped_mask])
    cyc_token_index, cyc_token_contour = match_token(cyc_pos, cyc_valid_mask, agent_heading[cyclist_mask],
                                                            'cyc', agent_category[cyclist_mask],
                                                            matching_extra_mask[cyclist_mask])

    # token_index: [NA, 18(90/5)] 每个代理在90帧中匹配到的18个token索引
    token_index = torch.zeros((agent_pos.shape[0], veh_token_index.shape[1])).to(torch.int64)
    token_index[vehicle_mask] = veh_token_index
    token_index[ped_mask] = ped_token_index
    token_index[cyclist_mask] = cyc_token_index

    # token_contour: [NA, 18, 4, 2] 每个代理在90帧中匹配到的18个token对应的矩形信息
    token_contour = torch.zeros((agent_pos.shape[0], veh_token_contour.shape[1],
                                    veh_token_contour.shape[2], veh_token_contour.shape[3]))
    token_contour[vehicle_mask] = veh_token_contour
    token_contour[ped_mask] = ped_token_contour
    token_contour[cyclist_mask] = cyc_token_contour

    # trajectory_token_veh = torch.from_numpy(trajectory_token['veh']).clone().to(torch.float)
    # trajectory_token_ped = torch.from_numpy(trajectory_token['ped']).clone().to(torch.float)
    # trajectory_token_cyc = torch.from_numpy(trajectory_token['cyc']).clone().to(torch.float)

    # agent_token_traj = torch.zeros((agent_pos.shape[0], trajectory_token_veh.shape[0], 4, 2))
    # agent_token_traj[vehicle_mask] = trajectory_token_veh
    # agent_token_traj[ped_mask] = trajectory_token_ped
    # agent_token_traj[cyclist_mask] = trajectory_token_cyc

    if not training:
        token_valid_mask[matching_extra_mask, 1] = True

    data['agent']['token_idx'] = token_index            # [NA, 18]
    data['agent']['token_contour'] = token_contour      # [NA, 18, 4, 2]
    token_pos = token_contour.mean(dim=2)               
    data['agent']['token_pos'] = token_pos              # [NA, 18, 2]
    diff_xy = token_contour[:, :, 0, :] - token_contour[:, :, 3, :]
    data['agent']['token_heading'] = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])  # [NA, 18]
    data['agent']['agent_valid_mask'] = token_valid_mask                                # [NA, 18]

    vel = torch.cat([token_pos.new_zeros(data['agent']['num_nodes'], 1, 2),
                        ((token_pos[:, 1:] - token_pos[:, :-1]) / (0.1 * shift))], dim=1)
    vel_valid_mask = torch.cat([torch.zeros(token_valid_mask.shape[0], 1, dtype=torch.bool),
                                (token_valid_mask * token_valid_mask.roll(shifts=1, dims=1))[:, 1:]], dim=1)
    vel[~vel_valid_mask] = 0
    vel[data['agent']['valid_mask'][:, current_step], 1] = data['agent']['velocity'][
                                                                data['agent']['valid_mask'][:, current_step],
                                                                current_step, :2]

    data['agent']['token_velocity'] = vel

    return data

def wrap_angle(
        angle: torch.Tensor,
        min_val: float = -math.pi,
        max_val: float = math.pi) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)

def interplating_polyline(polylines, heading, distance=0.5, split_distace=5):
    # 多段线切分长度为5米，多段线内部点之间距离为2.5米，即每条多段线由3个点构成
    # Calculate the cumulative distance along the path, up-sample the polyline to 0.5 meter
    dist_along_path_list = [[0]]
    polylines_list = [[polylines[0]]]
    for i in range(1, polylines.shape[0]):
        euclidean_dist = euclidean(polylines[i, :2], polylines[i - 1, :2])
        heading_diff = min(abs(max(heading[i], heading[i - 1]) - min(heading[1], heading[i - 1])),
                           abs(max(heading[i], heading[i - 1]) - min(heading[1], heading[i - 1]) + math.pi))
        if heading_diff > math.pi / 4 and euclidean_dist > 3:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        elif heading_diff > math.pi / 8 and euclidean_dist > 3:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        elif heading_diff > 0.1 and euclidean_dist > 3:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        elif euclidean_dist > 10:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        else:
            dist_along_path_list[-1].append(dist_along_path_list[-1][-1] + euclidean_dist)
            polylines_list[-1].append(polylines[i])
    # plt.plot(polylines[:, 0], polylines[:, 1])
    # plt.savefig('tmp.jpg')
    new_x_list = []
    new_y_list = []
    multi_polylines_list = []
    for idx in range(len(dist_along_path_list)):
        if len(dist_along_path_list[idx]) < 2:
            continue
        dist_along_path = np.array(dist_along_path_list[idx])
        polylines_cur = np.array(polylines_list[idx])
        # Create interpolation functions for x and y coordinates
        fx = interp1d(dist_along_path, polylines_cur[:, 0])
        fy = interp1d(dist_along_path, polylines_cur[:, 1])
        # fyaw = interp1d(dist_along_path, heading)

        # Create an array of distances at which to interpolate
        new_dist_along_path = np.arange(0, dist_along_path[-1], distance)
        new_dist_along_path = np.concatenate([new_dist_along_path, dist_along_path[[-1]]])
        # Use the interpolation functions to generate new x and y coordinates
        new_x = fx(new_dist_along_path)
        new_y = fy(new_dist_along_path)
        # new_yaw = fyaw(new_dist_along_path)
        new_x_list.append(new_x)
        new_y_list.append(new_y)

        # Combine the new x and y coordinates into a single array
        new_polylines = np.vstack((new_x, new_y)).T
        polyline_size = int(split_distace / distance)
        if new_polylines.shape[0] >= (polyline_size + 1):
            padding_size = (new_polylines.shape[0] - (polyline_size + 1)) % polyline_size
            final_index = (new_polylines.shape[0] - (polyline_size + 1)) // polyline_size + 1
        else:
            padding_size = new_polylines.shape[0]
            final_index = 0
        multi_polylines = None
        new_polylines = torch.from_numpy(new_polylines)
        new_heading = torch.atan2(new_polylines[1:, 1] - new_polylines[:-1, 1],
                                  new_polylines[1:, 0] - new_polylines[:-1, 0])
        new_heading = torch.cat([new_heading, new_heading[-1:]], -1)[..., None]
        new_polylines = torch.cat([new_polylines, new_heading], -1)
        if new_polylines.shape[0] >= (polyline_size + 1):
            multi_polylines = new_polylines.unfold(dimension=0, size=polyline_size + 1, step=polyline_size)
            multi_polylines = multi_polylines.transpose(1, 2)
            multi_polylines = multi_polylines[:, ::5, :]
        if padding_size >= 3:
            last_polyline = new_polylines[final_index * polyline_size:]
            last_polyline = last_polyline[torch.linspace(0, last_polyline.shape[0] - 1, steps=3).long()]
            if multi_polylines is not None:
                multi_polylines = torch.cat([multi_polylines, last_polyline.unsqueeze(0)], dim=0)
            else:
                multi_polylines = last_polyline.unsqueeze(0)
        if multi_polylines is None:
            continue
        multi_polylines_list.append(multi_polylines)
    if len(multi_polylines_list) > 0:
        multi_polylines_list = torch.cat(multi_polylines_list, dim=0)
    else:
        multi_polylines_list = None
    return multi_polylines_list

def tokenize_map(data):
    data['map_polygon']['type'] = data['map_polygon']['type'].to(torch.uint8)
    data['map_point']['type'] = data['map_point']['type'].to(torch.uint8)
    pt2pl = data[('map_point', 'to', 'map_polygon')]['edge_index']
    pt_type = data['map_point']['type'].to(torch.uint8)
    pt_side = torch.zeros_like(pt_type)
    pt_pos = data['map_point']['position'][:, :2]
    data['map_point']['orientation'] = wrap_angle(data['map_point']['orientation'])
    pt_heading = data['map_point']['orientation']
    split_polyline_type = []
    split_polyline_pos = []
    split_polyline_theta = []
    split_polyline_side = []
    pl_idx_list = []
    split_polygon_type = []
    data['map_point']['type'].unique()

    # 对多段线进行便利
    for i in sorted(np.unique(pt2pl[1])):
        # 每一条多段线对应的点
        index = pt2pl[0, pt2pl[1] == i]
        polygon_type = data['map_polygon']["type"][i]
        cur_side = pt_side[index]
        cur_type = pt_type[index]
        cur_pos = pt_pos[index]
        cur_heading = pt_heading[index]

        for side_val in np.unique(cur_side):
            for type_val in np.unique(cur_type):
                if type_val == 13:
                    continue
                indices = np.where((cur_side == side_val) & (cur_type == type_val))[0]
                if len(indices) <= 2:
                    continue
                split_polyline = interplating_polyline(cur_pos[indices].numpy(), cur_heading[indices].numpy())
                if split_polyline is None:
                    continue
                new_cur_type = cur_type[indices][0]
                new_cur_side = cur_side[indices][0]
                map_polygon_type = polygon_type.repeat(split_polyline.shape[0])
                new_cur_type = new_cur_type.repeat(split_polyline.shape[0])
                new_cur_side = new_cur_side.repeat(split_polyline.shape[0])
                cur_pl_idx = torch.Tensor([i])
                new_cur_pl_idx = cur_pl_idx.repeat(split_polyline.shape[0])
                split_polyline_pos.append(split_polyline[..., :2])
                split_polyline_theta.append(split_polyline[..., 2])
                split_polyline_type.append(new_cur_type)
                split_polyline_side.append(new_cur_side)
                pl_idx_list.append(new_cur_pl_idx)
                split_polygon_type.append(map_polygon_type)

    split_polyline_pos = torch.cat(split_polyline_pos, dim=0)
    split_polyline_theta = torch.cat(split_polyline_theta, dim=0)
    split_polyline_type = torch.cat(split_polyline_type, dim=0)
    split_polyline_side = torch.cat(split_polyline_side, dim=0)
    split_polygon_type = torch.cat(split_polygon_type, dim=0)
    pl_idx_list = torch.cat(pl_idx_list, dim=0)
    vec = split_polyline_pos[:, 1, :] - split_polyline_pos[:, 0, :]
    data['map_save'] = {}
    data['pt_token'] = {}
    data['map_save']['traj_pos'] = split_polyline_pos
    data['map_save']['traj_theta'] = split_polyline_theta[:, 0]  # torch.arctan2(vec[:, 1], vec[:, 0])
    data['map_save']['pl_idx_list'] = pl_idx_list
    data['pt_token']['type'] = split_polyline_type
    data['pt_token']['side'] = split_polyline_side
    data['pt_token']['pl_type'] = split_polygon_type
    data['pt_token']['num_nodes'] = split_polyline_pos.shape[0]
    return data

class CustomHeteroDataset(Dataset):
    def __init__(self, data_list):
        super(CustomHeteroDataset, self).__init__()
        self.data_list = data_list
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        batch_data = HeteroData()

        for node_type, node_data in self.data_list[idx].items():
            if isinstance(node_type, str):  # 处理节点数据
                if isinstance(node_data, dict):
                    for attr, value in node_data.items():
                        batch_data[node_type][attr] = value
                else:
                    batch_data[node_type] = [node_data]

        for edge_type, edge_data in self.data_list[idx].items():
            if isinstance(edge_type, tuple) and len(edge_type) == 3:  # 处理边数据
                if isinstance(edge_data, dict):
                    for attr, value in edge_data.items():
                        batch_data[edge_type][attr] = value
                else:
                    batch_data[edge_type] = edge_data
        return batch_data

def match_token_map(data):
    traj_pos = data['map_save']['traj_pos'].to(torch.float)
    traj_theta = data['map_save']['traj_theta'].to(torch.float)
    pl_idx_list = data['map_save']['pl_idx_list']
    token_sample_pt = map_token['sample_pt'].to(traj_pos.device)
    token_src = map_token['traj_src'].to(traj_pos.device)
    max_traj_len = map_token['traj_src'].shape[1]
    pl_num = traj_pos.shape[0]

    # 各地图多段线的起始点坐标xy
    pt_token_pos = traj_pos[:, 0, :].clone()
    # 各地图多段线的起始位置朝向
    pt_token_orientation = traj_theta.clone()
    # 将地图多段线由全局坐标系转换为局部坐标系
    cos, sin = traj_theta.cos(), traj_theta.sin()
    rot_mat = traj_theta.new_zeros(pl_num, 2, 2)
    rot_mat[..., 0, 0] = cos
    rot_mat[..., 0, 1] = -sin
    rot_mat[..., 1, 0] = sin
    rot_mat[..., 1, 1] = cos
    traj_pos_local = torch.bmm((traj_pos - traj_pos[:, 0:1]), rot_mat.view(-1, 2, 2))
    # 将坐标转换后的多段线与地图map_token进行匹配
    distance = torch.sum((token_sample_pt[None] - traj_pos_local.unsqueeze(1))**2, dim=(-2, -1))
    pt_token_id = torch.argmin(distance, dim=1)

    if noise:
        topk_indices = torch.argsort(torch.sum((token_sample_pt[None] - traj_pos_local.unsqueeze(1))**2, dim=(-2, -1)), dim=1)[:, :8]
        sample_topk = torch.randint(0, topk_indices.shape[-1], size=(topk_indices.shape[0], 1), device=topk_indices.device)
        pt_token_id = torch.gather(topk_indices, 1, sample_topk).squeeze(-1)

    cos, sin = traj_theta.cos(), traj_theta.sin()
    rot_mat = traj_theta.new_zeros(pl_num, 2, 2)
    rot_mat[..., 0, 0] = cos
    rot_mat[..., 0, 1] = sin
    rot_mat[..., 1, 0] = -sin
    rot_mat[..., 1, 1] = cos
    token_src_world = torch.bmm(token_src[None, ...].repeat(pl_num, 1, 1, 1).reshape(pl_num, -1, 2),
                                rot_mat.view(-1, 2, 2)).reshape(pl_num, token_src.shape[0], max_traj_len, 2) + traj_pos[:, None, [0], :]
    token_src_world_select = token_src_world.view(-1, 1024, 11, 2)[torch.arange(pt_token_id.view(-1).shape[0]), pt_token_id.view(-1)].view(pl_num, max_traj_len, 2)

    pl_idx_full = pl_idx_list.clone()
    token2pl = torch.stack([torch.arange(len(pl_idx_list), device=traj_pos.device), pl_idx_full.long()])
    count_nums = []
    for pl in pl_idx_full.unique():
        pt = token2pl[0, token2pl[1, :] == pl]
        left_side = (data['pt_token']['side'][pt] == 0).sum()
        right_side = (data['pt_token']['side'][pt] == 1).sum()
        center_side = (data['pt_token']['side'][pt] == 2).sum()
        count_nums.append(torch.Tensor([left_side, right_side, center_side]))
    # count_nums: [N_polyline, 3]分别记录每个原始多段线对应的左侧、右侧、中心token有多少
    count_nums = torch.stack(count_nums, dim=0)
    # 获取每个原始多段线对应的最多token数量
    max_token_num = int(count_nums.max().item())
    # 构建多段线的轨迹掩码 [N_polyline, 3, max_token_num]
    traj_mask = torch.zeros((int(len(pl_idx_full.unique())), 3, max_token_num), dtype=bool)
    idx_matrix = torch.arange(traj_mask.size(2)).unsqueeze(0).unsqueeze(0)
    idx_matrix = idx_matrix.expand(traj_mask.size(0), traj_mask.size(1), -1)    #[N_polyline, 3, max_token_num]
    counts_num_expanded = count_nums.unsqueeze(-1)                              #[N_polyline, 3, 1]
    traj_mask[idx_matrix < counts_num_expanded] = True

    data['pt_token']['traj_mask'] = traj_mask
    data['pt_token']['position'] = torch.cat([pt_token_pos, torch.zeros((data['pt_token']['num_nodes'], 1),
                                                                        device=traj_pos.device, dtype=torch.float)], dim=-1)
    data['pt_token']['orientation'] = pt_token_orientation
    data['pt_token']['height'] = data['pt_token']['position'][:, -1]
    data[('pt_token', 'to', 'map_polygon')] = {}
    data[('pt_token', 'to', 'map_polygon')]['edge_index'] = token2pl
    data['pt_token']['token_idx'] = pt_token_id
    return data

def sample_pt_pred(data):
    # traj_mask: [n_map_poly, 3, max_token_num]
    traj_mask = data['pt_token']['traj_mask']
    # 从每个原始多段线中随机选取1/3的traj值被掩码掉
    raw_pt_index = torch.arange(1, traj_mask.shape[2]).repeat(traj_mask.shape[0], traj_mask.shape[1], 1)
    masked_pt_index = raw_pt_index.view(-1)[torch.randperm(raw_pt_index.numel())[:traj_mask.shape[0]*traj_mask.shape[1]*((traj_mask.shape[2]-1)//3)].reshape(traj_mask.shape[0], traj_mask.shape[1], (traj_mask.shape[2]-1)//3)]
    masked_pt_index = torch.sort(masked_pt_index, -1)[0]
    # 有效掩码
    pt_valid_mask = traj_mask.clone()
    pt_valid_mask.scatter_(2, masked_pt_index, False)
    # 预测掩码
    pt_pred_mask = traj_mask.clone()
    pt_pred_mask.scatter_(2, masked_pt_index, False)
    tmp_mask = pt_pred_mask.clone()
    tmp_mask[:, :, :] = True
    tmp_mask.scatter_(2, masked_pt_index-1, False)
    pt_pred_mask.masked_fill_(tmp_mask, False)
    pt_pred_mask = pt_pred_mask * torch.roll(traj_mask, shifts=-1, dims=2)
    # 目标掩码
    pt_target_mask = torch.roll(pt_pred_mask, shifts=1, dims=2)
    # 通过traj_mask将生成的掩码向量从[n_map_poly, 3, max_token_num]转换为[n_polyline]的形式，使其与token信息对应
    data['pt_token']['pt_valid_mask'] = pt_valid_mask[traj_mask]
    data['pt_token']['pt_pred_mask'] = pt_pred_mask[traj_mask]
    data['pt_token']['pt_target_mask'] = pt_target_mask[traj_mask]

    return data

def plot_static_map(ax, batch):
    # 0:'DASH_SOLID_YELLOW', 1:'DASH_SOLID_WHITE', 2:'DASHED_WHITE', 3:'DASHED_YELLOW', 4:'DOUBLE_SOLID_YELLOW', 5:'DOUBLE_SOLID_WHITE', 6:'DOUBLE_DASH_YELLOW', 7:'DOUBLE_DASH_WHITE',
    # 8:'SOLID_YELLOW', 9:'SOLID_WHITE', 10:'SOLID_DASH_WHITE', 11:'SOLID_DASH_YELLOW', 12:'EDGE', 13:'NONE', 14:'UNKNOWN', 15:'CROSSWALK', 16:'CENTERLINE'
    _line_style = [['--', 2, 'yellow'], ['--', 2, 'grey'], ['--', 2, 'grey'], ['--', 2, 'yellow'], ['-', 2, 'yellow'], ['-', 2, 'grey'], ['--', 2, 'yellow'], ['--', 2, 'grey'],
                ['-', 2, 'yellow'], ['-', 2, 'grey'], ['--', 2, 'grey'], ['--', 2, 'yellow'], ['-', 3, 'black'], [], [], [':', 2, 'blue'], []]
    _center_colors = ['lightcoral', 'lightgreen', 'lightyellow', 'lightgray']

    # 准备数据
    polylines = []
    polyline_type = []
    for i in range(batch['map_polygon']['num_nodes']):
        point_idx = batch[('map_point', 'to', 'map_polygon')]['edge_index'][0, batch[('map_point', 'to', 'map_polygon')]['edge_index'][1] == i]
        polylines.append(torch.gather(batch['map_point']['position'][:, :2], dim=0, index=point_idx[..., None].repeat(1, 2)))
        polyline_type.append(batch['map_point']['type'][point_idx[0]])

    # 绘制每条地图线段
    for idx, (type, data) in enumerate(zip(polyline_type, polylines)):
        x = data[:, 0].numpy()
        y = data[:, 1].numpy()
        if (type == 13 or type == 14):
            continue
        elif (type == 16):
            ax.plot(x, y, marker='', linestyle='-', linewidth=2, color=_center_colors[batch['map_polygon']['light_type'][idx]], alpha=0.5)
        else:
            ax.plot(x, y, marker='', linestyle=_line_style[type][0], linewidth=_line_style[type][1], color=_line_style[type][2], alpha=0.8)

    ax.set_aspect('equal')
    ax.set_title(f"Scene <{batch['scenario_id'][0][0]}>")

def cal_polygon_contour(x, y, theta, width, length):
    left_front_x = x + 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
    left_front_y = y + 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)
    left_front = np.column_stack((left_front_x, left_front_y))

    right_front_x = x + 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
    right_front_y = y + 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
    right_front = np.column_stack((right_front_x, right_front_y))

    right_back_x = x - 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
    right_back_y = y - 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
    right_back = np.column_stack((right_back_x, right_back_y))

    left_back_x = x - 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
    left_back_y = y - 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)
    left_back = np.column_stack((left_back_x, left_back_y))

    polygon_contour = np.concatenate(
        (left_front[:, None, :], right_front[:, None, :], right_back[:, None, :], left_back[:, None, :]), axis=1)

    return polygon_contour

def visualize_pred(batch, pred):
    fig, ax_map = plt.subplots(figsize=(20, 20))
    ax_agent = ax_map.twinx()

    plot_static_map(ax_map, batch)

    traj = torch.cat([batch['agent']['position'][:, :11, :2], pred['pred_traj']], dim=1)
    head = torch.cat([batch['agent']['heading'][:, :11], pred['pred_head']], dim=1)

    N, T, _ = traj.shape
    agent_traj_all = cal_polygon_contour(
        traj.view(-1, 2)[..., 0], 
        traj.view(-1, 2)[..., 1], 
        head.view(-1), 
        batch['agent']['shape'].view(-1, 3)[..., 1], 
        batch['agent']['shape'].view(-1, 3)[..., 0]
    ).reshape(N, T, 4, 2)

    def update(frame):
        ax_agent.cla()
        ax_agent.axis('off')
        ax_agent.set_ylim(ax_map.get_ylim())
        polygons = []
        for agent_idx in range(agent_traj_all.shape[0]):
            polygon = patches.Polygon(agent_traj_all[agent_idx, frame], closed=True, fill='blue', edgecolor=None, alpha=0.9)  # fill=None 使其不填充
            ax_agent.add_patch(polygon)
            polygons.append(polygon)
        return polygons

    ani = FuncAnimation(fig, update, frames=np.arange(T), blit=True)

    # ani.save(f"/home/yangyh408/codes/SMART/data/limsim/{batch.scenario_id[0][0]}.gif", writer=PillowWriter(fps=10))
    plt.show()

def reference(data, visualize=False):
    token_data = tokenize_agent(data)
    token_data = tokenize_map(token_data)
    del token_data['city']
    if 'polygon_is_intersection' in token_data['map_polygon']:
        print("delete polygon_is_intersection")
        del token_data['map_polygon']['polygon_is_intersection']
    if 'route_type' in data['map_polygon']:
        print("delete route_type")
        del token_data['map_polygon']['route_type']
    dataset = CustomHeteroDataset([token_data])
    loader = DataLoader(dataset, batch_size=1)
    batch = next(iter(loader))
    batch = match_token_map(batch)
    batch = sample_pt_pred(batch)
    batch['agent']['av_index'] += batch['agent']['ptr'][:-1]
    
    model.eval()
    with torch.no_grad():
        # pred = model(batch)
        pred = model.inference(batch)
    
    if visualize:
        visualize_pred(batch, pred)

    return batch['agent']['id'][0], torch.cat((pred['pred_traj'], pred['pred_head'][..., None]), dim=-1)


if __name__ == '__main__':
    with open("/home/yangyh408/codes/SMART/data/limsim/limsim_meta_inter_all_green.pkl", 'rb') as handle:
        data = pickle.load(handle)
    pred = reference(data, visualize=True)
    print(pred)
    

 