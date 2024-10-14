from argparse import ArgumentParser
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from smart.datasets.scalable_dataset import MultiDataset
from smart.model import SMART
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging
from torch_geometric.data import Batch

from pathlib import Path
import pickle
from waymo_open_dataset.protos import scenario_pb2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import os
import torch
import numpy as np
from waymo_open_dataset.utils.sim_agents import visualizations

gif_dir = "videos/20241001_epoch02"

def generate_gif(batch, pred):
    with open(rf"data/process/val_scenarios/{batch.scenario_id[0]}.pickle", "rb") as handle:
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

    # 保存为gif
    ani.save(f"{gif_dir}/{batch.scenario_id[0]}.gif", writer=PillowWriter(fps=10))

    # 如果你想保存为mp4格式，可以使用以下代码：
    # ani.save("trajectories_animation.mp4", writer='ffmpeg', fps=5)

if __name__ == '__main__':
    config = load_config_act("configs/validation/validation_scalable.yaml")

    data_config = config.Dataset
    val_dataset = {
        "scalable": MultiDataset,
    }[data_config.dataset](root=data_config.root, split='val',
                            raw_dir=data_config.val_raw_dir,
                            processed_dir=data_config.val_processed_dir,
                            transform=WaymoTargetBuilder(config.Model.num_historical_steps, config.Model.decoder.num_future_steps))
    dataloader = DataLoader(val_dataset, batch_size=data_config.batch_size, shuffle=False, num_workers=data_config.num_workers,
                            pin_memory=data_config.pin_memory, persistent_workers=True if data_config.num_workers > 0 else False)
    
    Predictor = SMART
    logger = Logging().log(level='DEBUG')
    
    for i, batch in enumerate(iter(dataloader)):
        if i >= 100:
            break
        if os.path.exists(f"{gif_dir}/{batch.scenario_id[0]}.gif"):
            continue
        model = Predictor(config.Model)
        model.load_params_from_file(filename="ckpt/20241001/epoch=02-step=1460985-val_loss=2.71.ckpt",
                                    logger=logger)
        print(i, batch.scenario_id[0])
        data = model.match_token_map(batch)
        data = model.sample_pt_pred(data)
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        pred = model.inference(data)

        generate_gif(batch, pred)
