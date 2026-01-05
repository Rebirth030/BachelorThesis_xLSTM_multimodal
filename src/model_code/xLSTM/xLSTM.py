"""xLSTM model wrapper."""

from typing import Optional, List
import torch.nn as nn

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

class xLSTM(nn.Module):
    """xLSTM block stack with linear input/output projections."""

    def __init__(self,
                 window_size: int,
                 n_features: int,
                 horizon: int,
                 dropout: float,
                 num_blocks: int,
                 slstm_at: Optional[List[int]] = None,
                 hidden_size: int = 128,
                 mlstm_conv1d_kernel_size = 4,
                 mlstm_qkv_proj_blocksize = 4,
                 mlstm_num_heads = 4,
                 mlstm_proj_factor = 2.0,
                 slstm_conv1d_kernel = 4,
                 slstm_num_heads = 4
                 ):
        """Initialize the xLSTM stack and projection layers.

        Args:
            window_size: Sequence length for each input window.
            n_features: Number of input features per time step.
            horizon: Prediction horizon length.
            dropout: Dropout rate used in the stack.
            num_blocks: Number of xLSTM blocks.
            slstm_at: Indices where sLSTM blocks are placed.
            hidden_size: Embedding dimension for projections.
            mlstm_conv1d_kernel_size: Kernel size for mLSTM convolution.
            mlstm_qkv_proj_blocksize: Block size for mLSTM QKV projection.
            mlstm_num_heads: Number of attention heads for mLSTM.
            mlstm_proj_factor: Projection factor for mLSTM.
            slstm_conv1d_kernel: Kernel size for sLSTM convolution.
            slstm_num_heads: Number of attention heads for sLSTM.
        """
        super().__init__()

        stack_cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=mlstm_conv1d_kernel_size,
                qkv_proj_blocksize=mlstm_qkv_proj_blocksize,
                num_heads=mlstm_num_heads,
                proj_factor=mlstm_proj_factor,
                embedding_dim=hidden_size,
            )),
            slstm_block=sLSTMBlockConfig(slstm=sLSTMLayerConfig(
                conv1d_kernel_size=slstm_conv1d_kernel,
                num_heads=slstm_num_heads,
                embedding_dim=hidden_size,
                dropout=dropout,
                backend="vanilla",
            ),),
            context_length=window_size,
            num_blocks=num_blocks,
            slstm_at=slstm_at,
            dropout=dropout,
            bias=True,
            embedding_dim=hidden_size,
        )

        self.input_projection = nn.Linear(n_features, hidden_size)
        self.xlstm_stack = xLSTMBlockStack(stack_cfg)
        self.output_projection = nn.Linear(hidden_size, horizon)



    def forward(self, x):
        """Run a forward pass through the xLSTM stack.

        Args:
            x: Input tensor shaped [B, S, F].

        Returns:
            Tensor of predictions shaped [B, H].
        """
        x = self.input_projection(x)
        x = self.xlstm_stack(x)
        x = self.output_projection(x[:, -1])
        return x
