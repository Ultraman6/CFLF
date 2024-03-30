import os
import pickle
from datetime import datetime


class TaskResultManager:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        # 确保保存目录存在，如果不存在，则创建它
        os.makedirs(save_dir, exist_ok=True)
        self.his_list = self.read_pkl_files()

    def set_save_dir(self, save_dir):
        self.save_dir = save_dir

    def save_task_result(self, task_name, result):
        # 格式化当前时间
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        # 构建文件名
        file_name = f"{timestamp}_{task_name}.pkl"
        # 完整的文件路径
        file_path = os.path.join(self.save_dir, file_name)
        # 保存结果到.pkl文件
        with open(file_path, 'wb') as file:
            pickle.dump(result, file)
        absolute_path = os.path.abspath(file_path)
        print(f"Saved: {absolute_path}")

    def read_pkl_files(self):
        absolute_path = os.path.abspath(self.save_dir)
        print(f"Read: {absolute_path}")
        task_details = []
        for file in os.listdir(self.save_dir):
            if file.endswith(".pkl"):
                # 解析文件名以获取时间和任务名
                timestamp, task_name = file.split('_', 1)
                task_name = task_name.rsplit('.', 1)[0]  # 移除.pkl后缀
                task_details.append({'time': timestamp, 'name': task_name, 'file_name': file})
        return task_details

    def load_task_result(self, file_name):
        file_path = os.path.join(self.save_dir, file_name)
        if os.path.exists(file_path) and file_name.endswith(".pkl"):
            with open(file_path, 'rb') as file:
                timestamp, task_name = file_path.split('_', 1)
                task_name = task_name.rsplit('.', 1)[0]  # 移除.pkl后缀
                task_details = {'time': timestamp, 'name': task_name, 'info': pickle.load(file)}
                return task_details
        else:
            raise FileNotFoundError(f"{file_name} not found in {self.save_dir}")

    def delete_task_result(self, file_name):
        file_path = os.path.join(self.save_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            absolute_path = os.path.abspath(file_path)
            print(f"Deleted: {absolute_path}")
        else:
            raise FileNotFoundError(f"{file_name} not found in {self.save_dir}")