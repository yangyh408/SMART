# 系统环境

+ 系统环境：Ubuntu 20.04.6 LTS

+ CUDA版本：CUDA Toolkits 11.6

+ python版本：3.9.20

+ pytorch版本：1.12.1+cu116


# 环境配置

+ 创建conda环境
    ```bash
    conda create -n traffic-ntp python=3.9
    ```

+ 安装pytorch
    > 下载链接：https://pytorch.org/get-started/previous-versions/ ，选择合适版本进行下载
    ```bash
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
    ```

+ 安装pytorch-geometric
    > 为避免安装失败，建议手动下载相关依赖库，下载链接：https://pytorch-geometric.com/whl/
    ```bash
    pip install torch_cluster-1.6.0+pt112cu116-cp39-cp39-linux_x86_64.whl 
    pip install torch_scatter-2.1.0+pt112cu116-cp39-cp39-linux_x86_64.whl
    pip install torch_sparse-0.6.16+pt112cu116-cp39-cp39-linux_x86_64.whl
    pip install torch_spline_conv-1.2.1+pt112cu116-cp39-cp39-linux_x86_64.whl

    pip install torch-geometric
    ```

+ 安装其他第三方库

    ```bash
    pip install -r requirements.txt
    ```

# 数据准备

1. 下载原始数据集

    > 下载链接：https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_2_0

    下载 Waymo Open Motion Dataset (`scenario protocol`)到data文件夹下，文件结构如下:
    ```
    TrafficNTP
    ├── data
    │   ├── waymo
    │   │   ├── training
    │   │   ├── validation
    │   │   ├── testing
    ```

    也可以通过软链接的形式将硬盘中的数据挂载到data文件夹下，如:
    ```bash
    ln -s /mnt/i/smart_data/womd_scenario_v_1_2_0/ ~/codes/SMART/waymo
    ``` 

2. 进行数据预处理
    
    运行下述代码分别对训练集、验证集、测试集进行文件预处理，并保存在`data/waymo_processed`目录下
    
    ```bash
    # 训练集
    python ./data_process/data_preprocess.py --input_dir ./data/waymo/training  --output_dir ./data/waymo_processed/training
    # 验证集
    python ./data_process/data_preprocess.py --input_dir ./data/waymo/validation  --output_dir ./data/waymo_processed/validation
    # 测试集
    python ./data_process/data_preprocess.py --input_dir ./data/waymo/testing  --output_dir ./data/waymo_processed/testing
    ```

    预处理后的文件目录如下：

    ```
    TrafficNTP
    ├── data
    │   ├── waymo
    │   │   ├── training
    │   │   ├── validation
    │   │   ├── testing
    │   ├── waymo_processed
    │   │   ├── training
    │   │   ├── validation
    │   │   ├── testing
    ```


# 模型训练
    
1. 修改配置文件`configs/train/train_scalable_local.yaml`

    + `Dataset/train_raw_dir`: **(必须)** 修改为预处理后的training文件夹路径

    + `Dataset/val_raw_dir`: **(必须)** 修改为预处理后的validation文件夹路径


2. 运行训练

    默认加载的配置文件为`./configs/train/train_scalable_local.yaml`, checkpoints的默认保存路径为`./ckpt`, 也可以通过在运行命令时添加参数来自定义训练信息，支持的参数如下：

    + `ckpt_path`: 指定加载预训练模型的模型参数和训练情况，用于继续训练（默认为空，即从头开始训练）

    + `config`: 指定训练配置文件的加载路径

    + `save_ckpt_path`: 指定checkpoints的保存路径


    ```bash
    # 加载默认配置进行训练
    python train.py

    # 指定配置进行训练
    python train.py --ckpt_path ${CKPT_PATH} --config ${CONFIG_PATH} --save_ckpt_path ${SAVE_DIR}
    ```

3. 查看训练情况

    在训练过程中或训练后可以通过tensorboard查看训练情况，训练日志保存在`lightning_logs`目录下，每次训练以不同的version来命名

    ```bash
    tensorboard --logdir=./lightning_logs/version_xxx 
    ```

# 模型推理

+ `jupyter/data_process.ipynb`

    该文件展示了模型从最开始加载WOMD数据集中的场景，经历一系列数据预处理过程后组织成batch数据喂入模型进行推理的全过程，并附有每一步操作的功能和对象的组织形式

+ `jupyter/split_inference.ipynb`

    该文件展示了如何通过pytorch-lightning加载数据和模型并进行推理预测，在文件中还详细介绍了如何对输出的预测信息进行真实性指标计算和可视化GIF输出

+ `inference.py`

    该文件对`split_inference.ipynb`中的内容进行了汇总，可以对指定数量的场景进行推理预测，输出每个场景的GIF图并计算合计的真实性指标

    ```bash
    # 默认对1000个验证集场景进行推理评价
    python inference.py
    
    # 指定场景数量进行推理评价
    python inference.py --scene_num 10
    ```

+ `jupyer/split_limsim_inference.ipynb`

    该文件展示了如何调用在LimSim中预处理完成的轨迹和代理数据，并将其用于仿真任务。关于LimSim环境中的代理和地图信息提取过程，详见`MARL-LimSim/map_parse.ipynb`

+ `limsim_inference.py`

    该文件对`split_limsim_inference.ipynb`中的内容进行了汇总，并提供了`reference`接口供LimSim环境调用



