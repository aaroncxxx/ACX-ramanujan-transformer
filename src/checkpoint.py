"""
检查点管理模块 (v1.6)

完善断点续训：保存/恢复模型权重、优化器状态、学习率调度器状态、
训练步数、随机种子、scaler 状态（混合精度）。
"""

import os
import random
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger('acx_ramanujan')


class CheckpointManager:
    """
    检查点管理器 (v1.6)

    功能:
        - 保存完整训练状态（模型、优化器、调度器、scaler、步数、种子）
        - 自动恢复训练
        - 保留最近 N 个检查点（避免磁盘占满）
    """

    def __init__(self, checkpoint_dir: str, max_keep: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        epoch: int = 0,
        global_step: int = 0,
        best_val_loss: float = float('inf'),
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        保存完整训练检查点

        Returns:
            保存路径
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'best_val_loss': best_val_loss,
            'random_state': random.getstate(),
            'numpy_state': None,
            'torch_state': torch.random.get_rng_state(),
            'cuda_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()

        try:
            import numpy as np
            checkpoint['numpy_state'] = np.random.get_state()
        except ImportError:
            pass

        if extra:
            checkpoint['extra'] = extra

        filename = f"checkpoint_step{global_step:08d}.pt"
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)

        logger.info(f"检查点已保存: {filepath} (step={global_step}, val_loss={best_val_loss:.4f})")

        # 清理旧检查点
        self._cleanup()

        return str(filepath)

    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        path: Optional[str] = None,
        device: str = 'cpu',
    ) -> Dict[str, Any]:
        """
        加载检查点

        Args:
            model: 模型
            optimizer: 优化器（可选，仅恢复训练时需要）
            scheduler: 学习率调度器（可选）
            scaler: GradScaler（可选）
            path: 检查点路径。None 则自动查找最新
            device: 加载设备

        Returns:
            包含 epoch, global_step, best_val_loss 等信息的字典
        """
        if path is None:
            path = self._find_latest()

        if path is None or not os.path.exists(path):
            logger.warning("未找到检查点，从头开始训练")
            return {'epoch': 0, 'global_step': 0, 'best_val_loss': float('inf')}

        logger.info(f"加载检查点: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # 恢复随机状态
        if 'random_state' in checkpoint and checkpoint['random_state'] is not None:
            random.setstate(checkpoint['random_state'])

        if 'torch_state' in checkpoint and checkpoint['torch_state'] is not None:
            torch.random.set_rng_state(checkpoint['torch_state'])

        if 'cuda_state' in checkpoint and checkpoint['cuda_state'] is not None:
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint['cuda_state'])

        try:
            import numpy as np
            if 'numpy_state' in checkpoint and checkpoint['numpy_state'] is not None:
                np.random.set_state(checkpoint['numpy_state'])
        except (ImportError, Exception):
            pass

        result = {
            'epoch': checkpoint.get('epoch', 0),
            'global_step': checkpoint.get('global_step', 0),
            'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        }

        if 'extra' in checkpoint:
            result['extra'] = checkpoint['extra']

        logger.info(f"恢复训练: epoch={result['epoch']}, step={result['global_step']}, "
                    f"best_val_loss={result['best_val_loss']:.4f}")

        return result

    def _find_latest(self) -> Optional[str]:
        """查找最新的检查点"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step*.pt"))
        if not checkpoints:
            return None
        return str(checkpoints[-1])

    def _cleanup(self):
        """保留最近 max_keep 个检查点"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step*.pt"))
        while len(checkpoints) > self.max_keep:
            old = checkpoints.pop(0)
            old.unlink()
            logger.debug(f"删除旧检查点: {old}")
