from typing import List
from dataclasses import asdict
from lasvsim_env.config import Config
from lasvsim_env.lasvsim_env_qianxing import LasvsimEnv

class CloudServer:
    occupied_record_indexs = []
    free_record_indexs = list(range(0, 10))  

    def __new__(cls, *args, **kwargs):
        if not cls.free_record_indexs:
            raise RuntimeError("No free record IDs available")
        instance = super().__new__(cls)
        record_index = cls.free_record_indexs.pop(0)
        cls.occupied_record_indexs.append(record_index)
        instance.record_index = record_index
        print(f"Creating instance with record_index: {record_index}")
        return instance

    def __del__(self):
        record_index = self.record_index
        type(self).occupied_record_indexs.remove(record_index)
        type(self).free_record_indexs.append(record_index)
        print(f"Deleting instance with record_index: {record_index}")

    def __init__(self):
        pass

    def init_qx(self, env_config: Config, qx_config:dict):
        env_config_dict = asdict(env_config)
        env_config = Config.from_partial_dict(env_config_dict)
        self.env = LasvsimEnv(**qx_config, env_config=env_config_dict)

    def reset_qx(self, options: List[dict] = None):
        return self.env.reset()
    
    def get_all_ref_param(self):
        return self.env.get_all_ref_param()
    
    def step_qx(self, action):
        return self.env.step(action)

    def close_qx(self):
        self.env.stop_remote_lasvsim()