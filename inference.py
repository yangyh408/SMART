from smart.datasets.scalable_dataset import MultiDataset
from smart.model import SMART
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging
from smart.metrics.real_metrics.custom_metrics import RealMetrics
from smart.metrics.real_metrics.real_features import compute_real_metric_features

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import os
import torch
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from waymo_open_dataset.utils.sim_agents import visualizations
from waymo_open_dataset.protos import scenario_pb2

OUTPUT_DIR = r"/media/yangyh408/4A259082626F01B9/output_20241021_epoch07"
SCENARIO_DIR = r"/media/yangyh408/4A259082626F01B9/womd_scenario_v_1_2_0/val_scenarios"
REAL_METRICS_DIR = os.path.join(OUTPUT_DIR, 'real_metrics')
GIF_DIR = os.path.join(OUTPUT_DIR, 'gifs')

def generate_gif(batch, pred):
    with open(os.path.join(SCENARIO_DIR, rf"{batch.scenario_id[0]}.pickle"), "rb") as handle:
        scenario = scenario_pb2.Scenario.FromString(bytes.fromhex(pickle.load(handle).hex()))
    N, T, _ = pred['pred_traj'].shape
    real_trajectory = pred['gt']
    pred_trajectory = pred['pred_traj']

    # 创建颜色表（每个物体使用不同颜色）
    colors = plt.cm.viridis(torch.linspace(0, 1, N))  # 使用viridis颜色表

    # 设置绘图环境
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title(f"scene {batch.scenario_id[0]}")
    visualizations.add_map(ax, scenario)

    alpha_values = np.linspace(0.1, 0.8, 8) 
    # 初始化真实轨迹和预测轨迹的绘制对象
    # real_lines = [ax.plot([], [], 'o', color=colors[i], markersize=3, alpha=0.3)[0] for i in range(N)]  # 真实轨迹
    pred_lines = [ax.plot([], [], 'o', color=colors[i], markersize=3, alpha=0.8)[0] for i in range(N)]  # 预测轨迹

    # 更新函数，每一帧调用一次
    def update(frame):
        for i in range(N):
            # 真实轨迹更新
            # real_lines[i].set_data(real_trajectory[i, frame, 0], real_trajectory[i, frame, 1])
            # 预测轨迹更新
            pred_lines[i].set_data(pred_trajectory[i, max(frame-8, 0):frame+1, 0], pred_trajectory[i, max(frame-8, 0): frame+1, 1])
        # return real_lines + pred_lines
        return pred_lines

    # 创建动画
    ani = FuncAnimation(fig, update, frames=T, interval=100, blit=True)

    # 如果你想保存为mp4格式，可以使用以下代码：
    # ani.save("trajectories_animation.mp4", writer='ffmpeg', fps=5)
    # 保存为gif
    ani.save(f"{GIF_DIR}/{batch.scenario_id[0]}.gif", writer=PillowWriter(fps=10))


def get_real_metrics(batch, pred):
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
    real_metrics.save(os.path.join(REAL_METRICS_DIR, f"{batch.scenario_id[0]}.pkl"))


if __name__ == '__main__':
    config = load_config_act("configs/validation/validation_scalable.yaml")

    data_config = config.Dataset
    val_dataset = {
        "scalable": MultiDataset,
    }[data_config.dataset](root=data_config.root, split='val',
                            raw_dir=data_config.val_raw_dir,
                            processed_dir=data_config.val_processed_dir,
                            transform=WaymoTargetBuilder(config.Model.num_historical_steps, config.Model.decoder.num_future_steps))
    dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=False, persistent_workers=True)
    
    pretrain_ckpt = "ckpt/20241021_1037/epoch=07-step=30440-val_loss=2.52.ckpt"
    Predictor = SMART
    logger = Logging().log(level='DEBUG')
    model = Predictor(config.Model)
    model.load_params_from_file(filename=pretrain_ckpt, logger=logger)
    model.eval()

    for i, batch in enumerate(tqdm(iter(dataloader), total=1000)):  
        if i >= 1000:
            break
        if os.path.exists(f"{GIF_DIR}/{batch.scenario_id[0]}.gif") and os.path.exists(f"{REAL_METRICS_DIR}/{batch.scenario_id[0]}.pkl"):
            continue

        with torch.no_grad():
            data = model.match_token_map(batch)
            data = model.sample_pt_pred(data)
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
            pred = model.inference(data)

        generate_gif(batch, pred)
        get_real_metrics(batch, pred)
