# 使用雪花算法生成唯一ID
import time

class SnowflakeIDGenerator:
    def __init__(self, datacenter_id, worker_id):
        self.epoch = int(time.mktime(time.strptime('2023-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')))
        self.datacenter_id = datacenter_id
        self.worker_id = worker_id
        self.sequence = 0

    def generate_id(self):
        current_time = int(time.time() * 1000) - self.epoch
        id_bits = (current_time << 22) | (self.datacenter_id << 17) | (self.worker_id << 12) | self.sequence
        self.sequence = (self.sequence + 1) & 0xFFF
        if self.sequence == 0:
            # 当前毫秒内生成的ID超过4095个，等待下一毫秒
            while (current_time == (int(time.time() * 1000) - self.epoch)):
                pass
        return id_bits
