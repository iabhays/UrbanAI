"""
Device management for GPU/CPU inference.

Manages device selection, health monitoring, and TensorRT optimization.
"""

import torch
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime
from loguru import logger


@dataclass
class DeviceInfo:
    """Device information."""
    device_id: str
    device_type: str  # "cuda", "cpu"
    name: str
    memory_total_mb: float
    memory_used_mb: float
    memory_free_mb: float
    utilization_percent: Optional[float] = None
    temperature_celsius: Optional[float] = None
    is_healthy: bool = True


class DeviceManager:
    """
    Device manager for inference.
    
    Manages GPU/CPU devices, health monitoring, and device selection.
    """
    
    def __init__(self, preferred_device: Optional[str] = None):
        """
        Initialize device manager.
        
        Args:
            preferred_device: Preferred device (e.g., "cuda:0", "cpu")
        """
        self.preferred_device = preferred_device
        self.available_devices: List[DeviceInfo] = []
        self.current_device: Optional[torch.device] = None
        self._discover_devices()
    
    def _discover_devices(self):
        """Discover available devices."""
        # CPU is always available
        cpu_info = DeviceInfo(
            device_id="cpu",
            device_type="cpu",
            name="CPU",
            memory_total_mb=0.0,
            memory_used_mb=0.0,
            memory_free_mb=0.0
        )
        self.available_devices.append(cpu_info)
        
        # Discover CUDA devices
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_info = self._get_cuda_device_info(i)
                self.available_devices.append(device_info)
        
        logger.info(f"Discovered {len(self.available_devices)} devices")
    
    def _get_cuda_device_info(self, device_id: int) -> DeviceInfo:
        """Get CUDA device information."""
        try:
            props = torch.cuda.get_device_properties(device_id)
            memory_total = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 2)
            memory_allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
            memory_reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 2)
            memory_free = memory_total - memory_reserved
            
            # Try to get utilization (requires nvidia-ml-py)
            utilization = None
            temperature = None
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                pass
            
            return DeviceInfo(
                device_id=f"cuda:{device_id}",
                device_type="cuda",
                name=props.name,
                memory_total_mb=memory_total,
                memory_used_mb=memory_reserved,
                memory_free_mb=memory_free,
                utilization_percent=utilization,
                temperature_celsius=temperature,
                is_healthy=self._check_device_health(device_id)
            )
        except Exception as e:
            logger.error(f"Failed to get CUDA device info: {e}")
            return DeviceInfo(
                device_id=f"cuda:{device_id}",
                device_type="cuda",
                name="Unknown",
                memory_total_mb=0.0,
                memory_used_mb=0.0,
                memory_free_mb=0.0,
                is_healthy=False
            )
    
    def _check_device_health(self, device_id: int) -> bool:
        """Check if device is healthy."""
        try:
            # Simple health check: try to allocate small tensor
            test_tensor = torch.zeros(1, device=f"cuda:{device_id}")
            del test_tensor
            torch.cuda.empty_cache()
            return True
        except Exception:
            return False
    
    def get_device(self, device_id: Optional[str] = None) -> torch.device:
        """
        Get device for inference.
        
        Args:
            device_id: Specific device ID (None for auto-select)
        
        Returns:
            PyTorch device
        """
        if device_id:
            device = torch.device(device_id)
        elif self.preferred_device:
            device = torch.device(self.preferred_device)
        elif torch.cuda.is_available():
            # Select best GPU
            best_device = self._select_best_gpu()
            device = torch.device(best_device) if best_device else torch.device("cpu")
        else:
            device = torch.device("cpu")
        
        self.current_device = device
        return device
    
    def _select_best_gpu(self) -> Optional[str]:
        """Select best GPU based on available memory."""
        gpu_devices = [d for d in self.available_devices if d.device_type == "cuda" and d.is_healthy]
        
        if not gpu_devices:
            return None
        
        # Select GPU with most free memory
        best = max(gpu_devices, key=lambda d: d.memory_free_mb)
        return best.device_id
    
    def get_device_info(self, device_id: Optional[str] = None) -> Optional[DeviceInfo]:
        """
        Get device information.
        
        Args:
            device_id: Device ID (None for current device)
        
        Returns:
            Device information
        """
        if device_id is None:
            device_id = str(self.current_device) if self.current_device else "cpu"
        
        for device_info in self.available_devices:
            if device_info.device_id == device_id:
                # Update info
                if device_info.device_type == "cuda":
                    device_id_int = int(device_id.split(":")[1])
                    return self._get_cuda_device_info(device_id_int)
                return device_info
        
        return None
    
    def monitor_health(self) -> Dict[str, DeviceInfo]:
        """
        Monitor health of all devices.
        
        Returns:
            Dictionary of device info by device ID
        """
        health_status = {}
        
        for device_info in self.available_devices:
            if device_info.device_type == "cuda":
                device_id_int = int(device_info.device_id.split(":")[1])
                updated_info = self._get_cuda_device_info(device_id_int)
                health_status[device_info.device_id] = updated_info
            else:
                health_status[device_info.device_id] = device_info
        
        return health_status


class TensorRTEngine:
    """
    TensorRT optimization engine wrapper.
    
    Provides interface for TensorRT optimization (placeholder for full implementation).
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize TensorRT engine.
        
        Args:
            model_path: Path to TensorRT engine file
        """
        self.model_path = model_path
        self.engine = None
        self.is_available = self._check_tensorrt_availability()
    
    def _check_tensorrt_availability(self) -> bool:
        """Check if TensorRT is available."""
        try:
            import tensorrt as trt
            return True
        except ImportError:
            logger.warning("TensorRT not available. Install tensorrt package for GPU optimization.")
            return False
    
    def build_engine(
        self,
        onnx_model_path: str,
        output_path: str,
        precision: str = "FP16",
        workspace_size: int = 1073741824  # 1GB
    ) -> bool:
        """
        Build TensorRT engine from ONNX model.
        
        Args:
            onnx_model_path: Path to ONNX model
            output_path: Path to save TensorRT engine
            precision: Precision mode (FP32, FP16, INT8)
            workspace_size: Workspace size in bytes
        
        Returns:
            True if successful
        """
        if not self.is_available:
            logger.warning("TensorRT not available, cannot build engine")
            return False
        
        # Placeholder for TensorRT engine building
        # In production, implement full TensorRT builder
        logger.info(f"TensorRT engine building placeholder for {onnx_model_path}")
        logger.info(f"Would build {precision} engine with {workspace_size} workspace")
        
        return False  # Placeholder
    
    def load_engine(self, engine_path: str) -> bool:
        """
        Load TensorRT engine.
        
        Args:
            engine_path: Path to TensorRT engine
        
        Returns:
            True if successful
        """
        if not self.is_available:
            return False
        
        # Placeholder
        logger.info(f"TensorRT engine loading placeholder for {engine_path}")
        return False
    
    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run inference with TensorRT engine.
        
        Args:
            input_tensor: Input tensor
        
        Returns:
            Output tensor
        """
        if self.engine is None:
            raise RuntimeError("TensorRT engine not loaded")
        
        # Placeholder
        raise NotImplementedError("TensorRT inference not implemented")
