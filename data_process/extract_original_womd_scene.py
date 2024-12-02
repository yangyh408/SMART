import os
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
import tensorflow as tf
import pickle
from waymo_open_dataset.protos import scenario_pb2

out_dir = Path("/media/yangyh408/4A259082626F01B9/smart_waymo_processed/val_scenarios")

tfrecord_files = sorted([p.as_posix() for p in Path("/media/yangyh408/4A259082626F01B9/womd_scenario_v_1_2_0/validation").glob("*")])
assert len(tfrecord_files) == 150, "Validation tfrecord not complete, please dowload womd_scenario_v_1_2_0!"
tf.config.set_visible_devices([], "GPU")
tf_dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type="")
dataset_iterator = tf_dataset.as_numpy_iterator()
for i, scenario_bytes in enumerate(dataset_iterator):
    scenario = scenario_pb2.Scenario.FromString(bytes.fromhex(scenario_bytes.hex()))
    with open(out_dir / f"{scenario.scenario_id}.pickle", "wb") as handle:
        pickle.dump(scenario_bytes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(i)