"""每个逻辑门 / 每棵逻辑树从前一层哪些位置取输入。 Connectivity helpers for dense and convolutional logic layers."""

import torch
from torch import Tensor


def build_connectivity(input_dim: int, output_dim: int, seed: int) -> Tensor:
    """从[0, input_dim)中均匀采样两个输入索引，每个输出神经元采样两个输入索引。用于普通 dense logic layer !!!!!!!!!

    Args:
        input_dim: size of the previous layer (or input features).
        output_dim: number of neurons in this layer. 
        seed: RNG seed for reproducibility.

    Returns:
        LongTensor of shape [output_dim, 2]. 形状为[输出神经元数, 2]，每个神经元采样两个输入索引
    """
    g = torch.Generator() # 生成器，用于生成随机数
    g.manual_seed(int(seed)) # 设置随机种子，便于复现
    idx = torch.randint(
        low=0,
        high=input_dim,
        size=(output_dim, 2), # 形状为[输出神经元数, 2]，每个神经元采样从输入维度中采样两个输入索引
        generator=g, # 生成器，用于生成随机数
        dtype=torch.long,
    ) # 生成随机数，形状为[输出神经元数, 2]，每个神经元采样两个输入索引
    return idx

# 为每一棵卷积逻辑树随机决定：它从当前局部 receptive field 里的哪些位置取 8 个叶子输入。
def build_local_tree_leaf_connectivity(
    in_channels: int,
    receptive_field_size: int,
    out_channels: int, # 这一层有多少棵卷积逻辑树
    seed: int,
    num_leaves: int = 8,
    unique: bool = True,
    channels_per_tree: int = 2, # 每棵卷积逻辑树有多少个输入通道
) -> Tensor:
    """
    对每一个输出通道，也就是每一棵卷积逻辑树，从局部 receptive field 中随机选择 num_leaves 个叶子输入。
    in_channels * receptive_field_size * receptive_field_size 即receptive field被展平为一维
    """
    if in_channels < 1:
        raise ValueError("in_channels must be >= 1")
    if receptive_field_size < 1:
        raise ValueError("receptive_field_size must be >= 1")
    if out_channels < 1:
        raise ValueError("out_channels must be >= 1")
    if num_leaves < 2:
        raise ValueError("num_leaves must be >= 2")

    if channels_per_tree < 1:
        raise ValueError("channels_per_tree must be >= 1")
    effective_channels_per_tree = min(int(channels_per_tree), int(in_channels))

    spatial = int(receptive_field_size) * int(receptive_field_size) # 每个通道的空间大小，例如 3*3=9
    g = torch.Generator()
    g.manual_seed(int(seed))
    all_idx: list[Tensor] = [] # 存储每个通道的叶子输入索引
    for _ in range(out_channels):
        chosen_channels = torch.randperm(in_channels, generator=g)[:effective_channels_per_tree] # 从输入通道中随机选择2个通道，例如[0, 1]
        channel_offsets = chosen_channels * spatial # 计算每个通道的偏移量，例如[0*9, 1*9] = [0, 9]
        valid_positions = (
            channel_offsets.view(-1, 1) + torch.arange(spatial, dtype=torch.long).view(1, -1)
        ).reshape(-1) # 将通道偏移和空间位置索引展平为一维，即生成这些通道中所有候选位置索引，例如[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

        if unique and valid_positions.numel() >= num_leaves: # 如果需要唯一性，并且候选位置索引数量大于等于叶子数量，则随机选择叶子数量个位置索引
            perm = torch.randperm(valid_positions.numel(), generator=g)[:num_leaves] # 在valid_positions所有可能位置中，随机排列索引，然后取前num_leaves个，保证采样的叶子索引唯一且随机。
            leaf_idx = valid_positions[perm]
        else:
            ridx = torch.randint(
                low=0,
                high=valid_positions.numel(), # 候选位置索引数量
                size=(num_leaves,),
                generator=g,
                dtype=torch.long,
            )
            leaf_idx = valid_positions[ridx]
        all_idx.append(leaf_idx) # 将每个通道的叶子输入索引存储起来

    return torch.stack(all_idx, dim=0).long() # 将所有通道的叶子输入索引堆叠起来，形状为[out_channels, num_leaves]
    # 例如 leaf_idx =
    #[
    #  [ 0,  4,  7,  9, 11, 13, 15, 17],   # tree/kernel 0 的 8 个叶子索引
    #  [ 1,  2,  5,  8, 10, 12, 14, 16],   # tree/kernel 1 的 8 个叶子索引
    #  [ 0,  3,  6,  9, 12, 15, 16, 17],   # tree/kernel 2 的 8 个叶子索引
    #  [ 2,  4,  6, 10, 11, 13, 14, 15],   # tree/kernel 3 的 8 个叶子索引
    #]

