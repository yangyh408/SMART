from smart.datasets.preprocess import TokenProcessor  
import os  
import pickle  
from tqdm import tqdm  # 导入tqdm库  
import shutil

raw_dir = "/ssdfs/datahome/tj24005/smart_waymo_processed"
process_dir = "/ssdfs/datahome/tj24005/smart_waymo_processed/process"
move_target_dir = "/share/home/tj24005/data/smart_waymo_processed"

files = []

with open("scripts/data_token_process.pkl", 'rb') as handle:
    tasks = pickle.load(handle)
task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))
# num_tasks = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))
files_to_process = tasks[task_id]
notify = [len(files_to_process) * (i+1) // 100 for i in range(0, 100, 1)]

token_processor = TokenProcessor(2048) 
 
# with tqdm(total=len(files_to_process), desc=f"Task {task_id} Processing files") as pbar:  
for i, (split, f) in enumerate(files_to_process):
    if i+1 in notify:
        print(f"Task {task_id} Processing {i+1}/{len(files_to_process)}({(i+1)/len(files_to_process)*100:.2f}%)")
    raw_file_path = os.path.join(raw_dir, split, f)
    process_file_path = os.path.join(process_dir, split, f)
    move_file_path = os.path.join(move_target_dir, split, f)

    if os.path.exists(raw_file_path):
        with open (raw_file_path, 'rb') as handle:
            raw_data = pickle.load(handle)
        process_data = token_processor.preprocess(raw_data)
        with open(process_file_path, 'wb') as handle:  
            pickle.dump(process_data, handle)
        try:  
            shutil.move(raw_file_path, move_file_path)
        except Exception as e:  
            print(f"移动文件时出错: {e}")
        
        # pbar.update(1)