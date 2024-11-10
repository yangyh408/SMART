conda create -n smart python=3.8
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install pytorch-lightning==2.0.2

# download package from website: https://pytorch-geometric.com/whl/
pip install torch_cluster-1.6.0+pt112cu113-cp38-cp38-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install torch_sparse-0.6.14-cp38-cp38-linux_x86_64.whl
pip install torch_spline_conv-1.2.1+pt112cu113-cp38-cp38-linux_x86_64.whl

pip install torch_geometric
pip install waymo-open-dataset-tf-2-12-0==1.6.4
pip install shapely
pip install easydict
pip install jupyter
