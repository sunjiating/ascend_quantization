#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量策略（IncrementalStrategy）：从最敏感的层开始，按比例还原为原始精度
适合精度差一点点的场景，避免一次性还原50%的算子
"""

import copy
import collections
import math
from amct_onnx.common.auto_calibration.auto_calibration_strategy_base import AutoCalibrationStrategyBase

STOP_FLAG = 'stop_flag'
ROLL_BACK_CONFIG = 'roll_back_config'


class IncrementalStrategy(AutoCalibrationStrategyBase):
    """
    增量策略：从最敏感的层开始，按比例还原为原始精度
    
    与 BinarySearch 策略的区别：
    - BinarySearch: 第一次还原50%，然后二分搜索
    - IncrementalStrategy: 从最敏感的层开始，每次还原一定比例的层，逐步增加
    
    优点：
    - 适合精度差一点点的场景
    - 可以精确控制还原的比例
    - 避免过度还原
    - 比逐个还原快得多
    """
    
    def __init__(self, step_ratio=0.05, min_step=1):
        """
        初始化增量策略
        
        Args:
            step_ratio: 每次还原的层数比例，默认为0.05（5%）
                       - 0.05: 每次还原5%的层（推荐，平衡速度和精度）
                       - 0.1: 每次还原10%的层（更快，但可能过度还原）
                       - 0.02: 每次还原2%的层（更精确，但更慢）
            min_step: 最小还原层数，即使比例计算后小于此值，也至少还原这么多层
                      默认为1，确保至少还原1个层
        """
        super(IncrementalStrategy, self).__init__()
        self.cos_dict = {}
        self.record = ""
        self.sorted_cos_dict_list = []
        self.sorted_cos_quant_dict = collections.OrderedDict()
        self.current_index = 0  # 当前还原到的层索引
        self.step_ratio = step_ratio  # 每次还原的层数比例
        self.min_step = min_step  # 最小还原层数
        self.total_layers = 0  # 总层数
        self.last_acc_result = {}
        
    def initialize(self, ranking_info):
        """
        初始化策略
        
        Args:
            ranking_info: 敏感度排序字典，key是层名，value是敏感度值
                         注意：敏感度值越小，表示对精度影响越小
        """
        self.cos_dict = ranking_info
        self.record = 'init'
        
        # 按敏感度值从小到大排序（与BinarySearch一致）
        # 注意：余弦相似度值越小，表示量化后与原始模型差异越大，对精度影响越大
        # 所以应该从列表前面（相似度值小的层）开始还原
        self.sorted_cos_dict_list = sorted(
            self.cos_dict.items(), key=lambda d: d[1])
        
        # 初始化：所有层都量化
        self.sorted_cos_quant_dict = self.init_sorted_cos_quant_dict()
        self.total_layers = len(self.sorted_cos_dict_list)
        self.current_index = 0  # 从第一个（最敏感的，相似度值最小的）开始
        
    def init_sorted_cos_quant_dict(self):
        """初始化排序后的量化字典，所有层都量化"""
        sorted_dict = collections.OrderedDict()
        for item in self.sorted_cos_dict_list:
            sorted_dict[item[0]] = True  # True表示量化，False表示还原
        return sorted_dict
    
    def update_quant_config(self, metric_eval):
        """
        根据精度评估结果更新量化配置
        
        Args:
            metric_eval (Tuple[bool, float]): (是否满足要求, 损失值)
        
        Returns:
            dict: 包含 STOP_FLAG 和 ROLL_BACK_CONFIG 的字典
        """
        stop_flag = False
        accuracy, _ = metric_eval
        
        # 如果精度满足要求，保存当前结果
        if accuracy:
            self.last_acc_result = copy.deepcopy(self.result)
        
        # 第一次精度不够时，从最敏感的层开始还原
        if self.record == 'init' and accuracy is False:
            # 计算本次要还原的层数（按比例）
            num_layers = self._calculate_step_size()
            self._roll_back_layers(num_layers)
            self.result[STOP_FLAG] = False
            self.result[ROLL_BACK_CONFIG] = copy.deepcopy(self.sorted_cos_quant_dict)
            self.record = 'noinit'
            return self.result
        
        # 如果精度仍不够，继续还原更多层
        if not accuracy:
            # 计算本次要还原的层数（按比例）
            num_layers = self._calculate_step_size()
            self._roll_back_layers(num_layers)
            # 检查是否已经还原了所有层
            if self.current_index >= len(self.sorted_cos_dict_list):
                stop_flag = True
                # 如果所有层都还原了还不满足要求，返回上次满足要求的结果
                if self.last_acc_result:
                    self.last_acc_result[STOP_FLAG] = stop_flag
                    return self.last_acc_result
                # 如果没有满足要求的结果，返回全部还原的配置
                self.result[STOP_FLAG] = stop_flag
                self.result[ROLL_BACK_CONFIG] = copy.deepcopy(self.sorted_cos_quant_dict)
                return self.result
        else:
            # 精度满足要求，停止搜索
            stop_flag = True
            self.result[STOP_FLAG] = stop_flag
            self.result[ROLL_BACK_CONFIG] = copy.deepcopy(self.sorted_cos_quant_dict)
            return self.result
        
        # 继续搜索
        self.result[STOP_FLAG] = stop_flag
        self.result[ROLL_BACK_CONFIG] = copy.deepcopy(self.sorted_cos_quant_dict)
        return self.result
    
    def _calculate_step_size(self):
        """
        计算本次要还原的层数（基于比例）
        
        Returns:
            int: 本次要还原的层数
        """
        # 计算剩余未还原的层数
        remaining_layers = len(self.sorted_cos_dict_list) - self.current_index
        
        if remaining_layers <= 0:
            return 0
        
        # 按比例计算要还原的层数
        num_layers = max(
            self.min_step,  # 至少还原 min_step 个层
            math.ceil(remaining_layers * self.step_ratio)  # 按比例计算，向上取整
        )
        
        # 确保不超过剩余层数
        num_layers = min(num_layers, remaining_layers)
        
        return num_layers
    
    def _roll_back_layers(self, num_layers):
        """
        还原指定数量的层（从最敏感的层开始）
        
        Args:
            num_layers: 要还原的层数
        """
        # 从前往后还原（从相似度值小的层开始，这些层对精度影响大）
        for _ in range(num_layers):
            if self.current_index < len(self.sorted_cos_dict_list):
                layer_name = self.sorted_cos_dict_list[self.current_index][0]
                self.sorted_cos_quant_dict[layer_name] = False  # False表示还原为原始精度
                self.current_index += 1
            else:
                break  # 所有层都已还原
