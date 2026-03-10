# models/minkloc.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from models.layers.pooling_wrapper import PoolingWrapper


class MinkLoc(torch.nn.Module):
    def __init__(self, backbone: nn.Module, pooling: PoolingWrapper,
                 normalize_embeddings: bool = False, slice_branch: nn.Module = None):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.normalize_embeddings = normalize_embeddings
        self.slice_branch = slice_branch  # 精流切片分支
        self.stats = {}

    def forward(self, batch):
        # 判断当前是混合双流模式(Hybrid)还是单一流模式
        is_hybrid = 'coarse_coords' in batch and 'fine_coords' in batch

        if is_hybrid:
            # =========================================================
            # 混合双流模式: 构建两个独立的 SparseTensor
            # =========================================================
            coarse_x = ME.SparseTensor(batch['coarse_features'], coordinates=batch['coarse_coords'])
            fine_x = ME.SparseTensor(batch['fine_features'], coordinates=batch['fine_coords'])

            # 用于精流切片提取的 batch_size 以 fine_coords 为准
            batch_size = int(batch['fine_coords'][:, 0].max().item() + 1) if len(batch['fine_coords']) > 0 else 1
        else:
            # =========================================================
            # 单一流模式 (兼容老代码)
            # =========================================================
            coarse_x = ME.SparseTensor(batch['features'], coordinates=batch['coords'])
            fine_x = coarse_x  # 若只有一个流，精流和粗流共用张量

            batch_size = int(batch['coords'][:, 0].max().item() + 1) if len(batch['coords']) > 0 else 1

        # =========================================================
        # 流1 (粗流 Coarse): 传统的全局特征提取 (256-D)
        # =========================================================
        coarse_out = self.backbone(coarse_x)
        assert coarse_out.shape[
                   1] == self.pooling.in_dim, f'Backbone output tensor has: {coarse_out.shape[1]} channels. Expected: {self.pooling.in_dim}'

        coarse_out = self.pooling(coarse_out)
        if hasattr(self.pooling, 'stats'):
            self.stats.update(self.pooling.stats)

        assert coarse_out.dim() == 2, f'Expected 2-dimensional tensor (batch_size,output_dim). Got {coarse_out.dim()} dimensions.'
        assert coarse_out.shape[
                   1] == self.pooling.output_dim, f'Output tensor has: {coarse_out.shape[1]} channels. Expected: {self.pooling.output_dim}'

        if self.normalize_embeddings:
            coarse_out = F.normalize(coarse_out, dim=1)

        output = {'global': coarse_out}

        # =========================================================
        # 流2 (精流 Fine): 平行的切片序列提取 (64 * N-D)
        # =========================================================
        if self.slice_branch is not None:
            fine_seq = self.slice_branch(fine_x, batch_size)
            output['sequence'] = fine_seq

        return output

    def print_info(self):
        print('Model class: MinkLoc (Dual-Stream / Hybrid)')
        n_params = sum([param.nelement() for param in self.parameters()])
        print(f'Total parameters: {n_params}')
        n_params = sum([param.nelement() for param in self.backbone.parameters()])
        print(f'Backbone: {type(self.backbone).__name__} #parameters: {n_params}')
        n_params = sum([param.nelement() for param in self.pooling.parameters()])
        print(f'Pooling method: {self.pooling.pool_method}   #parameters: {n_params}')

        # 打印精流分支信息
        if self.slice_branch is not None:
            n_params = sum([param.nelement() for param in self.slice_branch.parameters()])
            print(f'Slice Branch: {type(self.slice_branch).__name__}   #parameters: {n_params}')
            print(f'# sequence slices: {self.slice_branch.num_slices}')
            print(f'# sequence feature dim: {self.slice_branch.feature_dim}')

        print('# channels from the backbone: {}'.format(self.pooling.in_dim))
        print('# output channels : {}'.format(self.pooling.output_dim))
        print(f'Embedding normalization: {self.normalize_embeddings}')