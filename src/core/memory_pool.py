"""
ATLAS Memory Optimization
内存优化工具

1. Numpy数组对象池 - 减少重复分配
2. 空间对象的__slots__优化
3. 预分配缓冲区
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque


class ArrayPool:
    """
    Numpy数组对象池

    避免频繁的数组分配/释放，减少GC压力。
    """

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._pools: Dict[Tuple, deque] = {}

    def _get_key(self, shape: Tuple, dtype: np.dtype) -> Tuple:
        """生成池键"""
        return (shape, dtype)

    def acquire(self, shape: Tuple, dtype: np.dtype = np.float64,
                default_value: Optional[float] = None) -> np.ndarray:
        """获取数组（从池或新分配）"""
        key = self._get_key(shape, dtype)

        if key in self._pools and self._pools[key]:
            arr = self._pools[key].popleft()
            if default_value is not None:
                arr.fill(default_value)
            return arr

        # 池中没有，新分配
        arr = np.empty(shape, dtype=dtype)
        if default_value is not None:
            arr.fill(default_value)
        return arr

    def release(self, arr: np.ndarray):
        """释放数组回池"""
        if arr is None:
            return

        key = self._get_key(arr.shape, arr.dtype)

        if key not in self._pools:
            self._pools[key] = deque(maxlen=self.max_size)

        self._pools[key].append(arr)

    def clear(self):
        """清空池"""
        self._pools.clear()


class WorkspacePool:
    """
    求解器工作区对象池

    为A*等算法预分配g_score、f_score等字典/列表
    """

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._dict_pool: deque = deque(maxlen=max_size)
        self._list_pool: deque = deque(maxlen=max_size)

    def acquire_dict(self) -> dict:
        """获取字典"""
        if self._dict_pool:
            d = self._dict_pool.popleft()
            d.clear()
            return d
        return {}

    def release_dict(self, d: dict):
        """释放字典"""
        if d is not None:
            d.clear()
            self._dict_pool.append(d)

    def acquire_list(self) -> list:
        """获取列表"""
        if self._list_pool:
            lst = self._list_pool.popleft()
            lst.clear()
            return lst
        return []

    def release_list(self, lst: list):
        """释放列表"""
        if lst is not None:
            lst.clear()
            self._list_pool.append(lst)

    def clear(self):
        """清空池"""
        self._dict_pool.clear()
        self._list_pool.clear()


# 全局池实例
_global_array_pool = ArrayPool(max_size=200)
_global_workspace_pool = WorkspacePool(max_size=50)


def get_array_pool() -> ArrayPool:
    """获取全局数组池"""
    return _global_array_pool


def get_workspace_pool() -> WorkspacePool:
    """获取全局工作区池"""
    return _global_workspace_pool


def reset_pools():
    """重置所有池"""
    _global_array_pool.clear()
    _global_workspace_pool.clear()
