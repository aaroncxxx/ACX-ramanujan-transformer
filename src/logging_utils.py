"""
训练日志模块 (v1.6)

支持 WandB / TensorBoard / none 三种日志后端。
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger('acx_ramanujan')


class TrainingLogger:
    """
    统一日志接口 (v1.6)

    支持:
        - wandb: Weights & Biases 在线实验追踪
        - tensorboard: TensorBoard 本地可视化
        - none: 仅标准日志
    """

    def __init__(self, logger_type: str = 'none', project: str = 'acx-ramanujan',
                 run_name: Optional[str] = None, config: Optional[Dict] = None):
        self.logger_type = logger_type
        self._writer = None
        self._wandb = None

        if logger_type == 'wandb':
            self._init_wandb(project, run_name, config)
        elif logger_type == 'tensorboard':
            self._init_tensorboard()
        else:
            logger.info("日志模式: none (仅标准输出)")

    def _init_wandb(self, project: str, run_name: Optional[str],
                    config: Optional[Dict]):
        try:
            import wandb
            wandb.init(project=project, name=run_name, config=config or {})
            self._wandb = wandb
            logger.info("WandB 已初始化")
        except ImportError:
            logger.warning("wandb 未安装，回退到标准日志。安装: pip install wandb")
            self.logger_type = 'none'

    def _init_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter()
            logger.info("TensorBoard 已初始化")
        except ImportError:
            logger.warning("tensorboard 未安装，回退到标准日志。安装: pip install tensorboard")
            self.logger_type = 'none'

    def log(self, metrics: Dict[str, Any], step: int):
        """记录指标"""
        if self.logger_type == 'wandb' and self._wandb is not None:
            self._wandb.log(metrics, step=step)
        elif self.logger_type == 'tensorboard' and self._writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._writer.add_scalar(key, value, step)

    def log_histogram(self, tag: str, values, step: int):
        """记录直方图（梯度分布等）"""
        if self.logger_type == 'tensorboard' and self._writer is not None:
            self._writer.add_histogram(tag, values, step)
        elif self.logger_type == 'wandb' and self._wandb is not None:
            self._wandb.log({tag: self._wandb.Histogram(values)}, step=step)

    def log_model_info(self, model, step: int = 0):
        """记录模型信息"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        info = {
            'model/total_params': total_params,
            'model/trainable_params': trainable_params,
        }
        self.log(info, step)
        logger.info(f"模型参数: {total_params:,} (可训练: {trainable_params:,})")

    def close(self):
        """关闭日志"""
        if self._writer is not None:
            self._writer.close()
        if self._wandb is not None:
            self._wandb.finish()
