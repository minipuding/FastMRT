"""这一部分是用于实现tensor的计算机视觉方法， 如形态学运算等"""
import torch
from typing import Any
from torch.nn import functional as F


class FastmrtCV:

    @staticmethod
    def tensor_dilation(image: torch.Tensor, kernel_size: int = 9, padding: Any = None):
        if image.ndim < 3:
            for _ in range(3 - image.ndim):
                image = image.unsqueeze(0)
        if not padding:
            padding = kernel_size // 2
        return F.max_pool2d(image.float(), kernel_size=kernel_size, stride=1, padding=padding).bool()

    @staticmethod
    def tensor_erosion(image: torch.Tensor, kernel_size: int = 9, padding: Any = None):
        if image.ndim < 3:
            for _ in range(3 - image.ndim):
                image = image.unsqueeze(0)
        if not padding:
            padding = kernel_size // 2
        return ~(F.max_pool2d((~image).float(), kernel_size=kernel_size, stride=1, padding=padding)).bool()

    @staticmethod
    def tensor_opening(image: torch.Tensor, kernel_size: int = 9, padding: Any = None):
        if image.ndim < 3:
            for _ in range(3 - image.ndim):
                image = image.unsqueeze(0)
        if not padding:
            padding = kernel_size // 2
        result = ~(F.max_pool2d((~image).float(), kernel_size=kernel_size, stride=1, padding=padding)).bool()
        result = F.max_pool2d(result.float(), kernel_size=kernel_size, stride=1, padding=padding).bool()
        return result

    @staticmethod
    def tensor_closing(image: torch.Tensor, kernel_size: int = 9, padding: Any = None):
        if image.ndim < 3:
            for _ in range(3 - image.ndim):
                image = image.unsqueeze(0)
        if not padding:
            padding = kernel_size // 2
        result = F.max_pool2d(image.float(), kernel_size=kernel_size, stride=1, padding=padding).bool()
        result = ~(F.max_pool2d((~result).float(), kernel_size=kernel_size, stride=1, padding=padding).bool())
        return result
