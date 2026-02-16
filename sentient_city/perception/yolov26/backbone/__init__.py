"""Backbone architectures for YOLOv26."""

from .csp_darknet import CSPDarknet
from .efficientnet import EfficientNetBackbone

__all__ = ["CSPDarknet", "EfficientNetBackbone"]
