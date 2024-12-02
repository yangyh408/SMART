from smart.datasets.scalable_dataset import MultiDataset
from smart.model import SMART
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging
from smart.metrics.real_metrics.custom_metrics import MetricFeatures, RealMetrics
from smart.metrics.real_metrics.real_features import compute_real_metric_features

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import os
import torch
import pickle
import dataclasses
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from argparse import ArgumentParser

# 设置推理的场景数量
INFERENCE_SCENE_NUM = 3

root_dir = Path(__file__).resolve().parent
real_metrics_dir = root_dir/'outputs/real_metrics'
if not os.path.exists(real_metrics_dir):
    os.makedirs(real_metrics_dir)
gif_dir = root_dir/'outputs/gifs'
if not os.path.exists(gif_dir):
    os.makedirs(gif_dir)

def generate_gif(batch, pred):
    # 为每个场景生成GIF图
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
        ax.set_title(f"Scene <{batch['scenario_id'][0]}>")

    fig, ax_map = plt.subplots(figsize=(20, 20))

    plot_static_map(ax_map, batch)

    N, T, _ = pred['pred_traj'].shape
    pred_trajectory = pred['pred_traj']

    # 创建颜色表（每个物体使用不同颜色）
    colors = plt.cm.viridis(torch.linspace(0, 1, N))  # 使用viridis颜色表

    pred_lines = [ax_map.plot([], [], 'o', color=colors[i], markersize=3, alpha=0.8)[0] for i in range(N)]

    # 更新函数，每一帧调用一次
    def update(frame):
        for i in range(N):
            pred_lines[i].set_data(pred_trajectory[i, max(frame-8, 0):frame+1, 0], pred_trajectory[i, max(frame-8, 0): frame+1, 1])
        return pred_lines

    # 创建动画
    ani = FuncAnimation(fig, update, frames=T, interval=100, blit=True)

    # 将图像保存为gif
    ani.save(gif_dir/f"{batch.scenario_id[0]}.gif", writer=PillowWriter(fps=10))

def get_real_metrics(batch, pred):
    # 提取每个场景的真实性指标
    log_info = {}
    log_info['x'] = batch['agent']['position'][:, 11:, 0]
    log_info['y'] = batch['agent']['position'][:, 11:, 1]
    log_info['heading'] = batch['agent']['heading'][:, 11:]
    log_info['length'] = batch['agent']['shape'][:, 11:, 0]
    log_info['width'] = batch['agent']['shape'][:, 11:, 1]
    log_info['valid'] = batch['agent']['valid_mask'][:, 11:]

    sim_info = {}
    sim_info['x'] = pred['pred_traj'][..., 0]
    sim_info['y'] = pred['pred_traj'][..., 1]
    sim_info['heading'] = pred['pred_head']
    sim_info['length'] = batch['agent']['shape'][:, 11:, 0]
    sim_info['width'] = batch['agent']['shape'][:, 11:, 1]
    sim_info['valid'] = batch['agent']['valid_mask'][:, 11:]

    real_metrics = RealMetrics()
    real_metrics.add_log_features(
        compute_real_metric_features(
            center_x = log_info['x'],
            center_y = log_info['y'],
            length = log_info['length'],
            width = log_info['width'],
            heading = log_info['heading'],
            valid = log_info['valid'],
        )
    )
    real_metrics.add_sim_features(
        compute_real_metric_features(
            center_x = sim_info['x'],
            center_y = sim_info['y'],
            length = sim_info['length'],
            width = sim_info['width'],
            heading = sim_info['heading'],
            valid = sim_info['valid'],
        )
    )
    real_metrics.compute_js_divergence(method='histogram', plot=False)
    real_metrics.save(real_metrics_dir/f"{batch.scenario_id[0]}.pkl")

def cal_total_real_metrics():
    # 汇总real_metrics_dir中的所有真实性指标给出合计数据值
    all_log_features = {}
    all_sim_features = {}
    for field in dataclasses.fields(MetricFeatures):
        all_log_features[field.name] = tf.zeros((0, 80), dtype=tf.float32)
        all_sim_features[field.name] = tf.zeros((0, 80), dtype=tf.float32)

    scene_num = len(os.listdir(real_metrics_dir))
    
    for file in tqdm(os.listdir(real_metrics_dir), desc="Processing files"):
        with open(os.path.join(real_metrics_dir, file), 'rb') as handle:
            data = pickle.load(handle)

        for field, feature in all_log_features.items():
            all_log_features[field] = tf.concat([feature, getattr(data.log_features, field)], axis=0)

        for field, feature in all_sim_features.items():
            all_sim_features[field] = tf.concat([feature, getattr(data.sim_features[0], field)], axis=0)

    real_metrics = RealMetrics()
    real_metrics.config['linear_accel'] = {
        'min_val': -6.0,
        'max_val': 6.0,
        'num_bins': 15,
        'bar_width': 0.35,
    }
    real_metrics.config['yaw_speed'] = {
        'min_val': -0.314,
        'max_val': 0.314,
        'num_bins': 20,
        'bar_width': 0.01,
    }

    log_metrics = MetricFeatures(**all_log_features)
    real_metrics.add_log_features(log_metrics)
    sim_metrics = MetricFeatures(**all_sim_features)
    real_metrics.add_sim_features(sim_metrics)
    real_metrics.compute_js_divergence(method='histogram', plot=False)
    
    print("-" * 50)
    print(f"Real Metrics with {scene_num} scenes:")
    for k, v in real_metrics.js_divergence.items():
        print(f"- {k}: {v: .3f}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scene_num', type=int, default=1000)
    args = parser.parse_args()

    config = load_config_act(root_dir/"configs/validation/validation_scalable.yaml")

    data_config = config.Dataset
    val_dataset = {
        "scalable": MultiDataset,
    }[data_config.dataset](root=data_config.root, split='val',
                            raw_dir=data_config.val_raw_dir,
                            processed_dir=data_config.val_processed_dir,
                            transform=WaymoTargetBuilder(config.Model.num_historical_steps, config.Model.decoder.num_future_steps))
    dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=False, persistent_workers=True)
    
    pretrain_ckpt = root_dir/"ckpt/20241021_1037/epoch=07-step=30440-val_loss=2.52.ckpt"
    Predictor = SMART
    logger = Logging().log(level='DEBUG')
    model = Predictor(config.Model)
    model.load_params_from_file(filename=pretrain_ckpt, logger=logger)
    model.eval()

    for i, batch in enumerate(tqdm(iter(dataloader), total=args.scene_num)):  
        if i >= args.scene_num:
            break
        if os.path.exists(f"{gif_dir}/{batch.scenario_id[0]}.gif") and os.path.exists(f"{real_metrics_dir}/{batch.scenario_id[0]}.pkl"):
            continue

        with torch.no_grad():
            data = model.match_token_map(batch)
            data = model.sample_pt_pred(data)
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
            pred = model.inference(data)

        generate_gif(batch, pred)
        get_real_metrics(batch, pred)

    cal_total_real_metrics()
