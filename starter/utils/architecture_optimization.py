"""
Architecture optimization utilities for hardware-aware model optimization in medical imaging.

This module provides comprehensive implementations of modern neural network optimization
techniques specifically designed for clinical deployment scenarios. Focuses on reducing
computational overhead, memory usage, and inference latency while maintaining diagnostic
accuracy for the PneumoniaMNIST binary classification task.

Key optimization strategies:
    - Interpolation Removal: Eliminates computational overhead from resolution upscaling
    - Depthwise Separable Convolutions: Reduces parameters and FLOPs significantly
    - Grouped Convolutions: Parallel channel processing for improved efficiency
    - Inverted Residual Blocks: Mobile-optimized residual architectures
    - Low-Rank Factorization: Matrix decomposition for parameter reduction
    - Channel Optimization: Memory layout and activation optimizations
    - Parameter Sharing: Weight reuse across similar layer configurations
"""

import copy
import math
from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def create_optimized_model(base_model: nn.Module, optimizations: Dict[str, Any]) -> nn.Module:
    """
    Apply selected optimization strategies in order to create a clinically-optimized model.

    Args:
        base_model: Original ResNet model to optimize for clinical deployment
        optimizations: Dictionary specifying which optimizations to apply with parameters:
            - 'interpolation_removal': bool - Remove upscaling overhead (recommended: True)
            - 'depthwise_separable': bool - Apply depthwise separable convolutions
            - 'grouped_conv': bool - Use grouped convolutions for parallel processing
            - 'channel_optimization': bool - Optimize memory layout and activations
            - 'inverted_residuals': bool - Replace blocks with inverted residuals
            - 'lowrank_factorization': bool - Apply matrix factorization to linear layers
            - 'parameter_sharing': bool - Share weights between similar layers
            
    Returns:
        Optimized model with selected techniques applied, ready for clinical deployment
        
    Example:
        >>> base_model = create_baseline_model()
        >>> optimization_config = {
        ...     'interpolation_removal': True,
        ...     'depthwise_separable': True,
        ...     'channel_optimization': True
        ... }
        >>> optimized_model = create_optimized_model(base_model, optimization_config)
        >>> print("Clinical deployment model ready")
    """
    model = copy.deepcopy(base_model)
  
    print("Starting clinical model optimization pipeline...")
    
    # Define a sensible, dependency‑aware optimization order
    # 1) Architectural changes that impact tensor shapes (interpolation/native input)
    # 2) Layer substitutions that change compute patterns (depthwise/grouped)
    # 3) Light hardware-aware tweaks (channels_last / in-place activations)
    # 4) Parameter-space reductions (low-rank, sharing)
    optimization_order = [
        'interpolation_removal',
        'depthwise_separable',
        'grouped_conv',
        'channel_optimization',
        'lowrank_factorization',
        'parameter_sharing',
        'inverted_residuals',  # optional; placed last due to larger structural change
    ]
    
    # Optimization function mapping - connects optimization names to their implementation
    # IMPORTANT: Make sure to experiment with different input parameters for each optimization function, if performance is suboptimal
    optimization_functions = {
        'interpolation_removal': lambda m: apply_interpolation_removal_optimization(m),
        'depthwise_separable': lambda m: apply_depthwise_separable_optimization(m),
        'grouped_conv': lambda m: apply_grouped_convolution_optimization(m),
        'channel_optimization': lambda m: apply_channel_optimization(m),
        'inverted_residuals': lambda m: apply_inverted_residual_optimization(m),
        'lowrank_factorization': lambda m: apply_lowrank_factorization(m),
        'parameter_sharing': lambda m: apply_parameter_sharing(m)
    }
    
    # Smart iteration through the defined optimization order
    applied_optimizations = []
    for opt_name in optimization_order:
        # Check if this optimization is requested and available
        if optimizations.get(opt_name, False) and opt_name in optimization_functions:
            print(f"   Applying {opt_name.replace('_', ' ')} optimization...")
            try:
                # Apply the optimization using the mapped function
                model = optimization_functions[opt_name](model)
                applied_optimizations.append(opt_name)
            except Exception as e:
                print(f"   ERROR: {opt_name} optimization failed: {e}")
        elif opt_name not in optimization_functions:
            print(f"   WARNING: Unknown optimization: {opt_name}")
    
    # Report results
    if applied_optimizations:
        print(f"Applied optimizations in order: {' → '.join(applied_optimizations)}")
    else:
        print("No optimizations were applied")
        
    return model

# --------------------------------------
# INTERPOLATION REMOVAL (NATIVE RESOLUTION)
# --------------------------------------

# TODO: Implement this optimization method, if selected in your optimization strategy
def apply_interpolation_removal_optimization(model: nn.Module, native_size: int = 64) -> nn.Module:
    """
    Remove interpolation overhead by processing images at native resolution.
    
    Args:
        model: Model with interpolation capability (e.g., ResNetBaseline)
        native_size: Native input resolution to process (64 for clinical deployment)
        
    Returns:
        Optimized model that processes at native resolution without interpolation

    Note: 
        In `data_loader.py`, we would also want to replace ImageNet stats with chest 
        X-ray specific to check if accuracy improves, but you can skip this for simplicity 
        as normalization affects accuracy/sensitivity and not operational efficiency.
        
    Example:
        >>> baseline_model = create_baseline_model()
        >>> optimized_model = apply_interpolation_removal_optimization(baseline_model, 64)
        >>> # Model now processes 64x64 images directly without upscaling
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)

    print(f"Applying native resolution optimization ({native_size}x{native_size})...")
    
    # TODO: Update the existing model class to bypasses interpolation and processes images at native resolution.
    # HINT: The ResNetBaseline model automatically interpolates input images from 64x64 to 224x224 
    # before passing them to the underlying ResNet. One option is to create a wrapper that:
    # 1. Stores the original model architecture and metadata
    # 2. Updates the input_size attribute to reflect native processing  
    # 3. In the forward pass, bypasses the interpolation step entirely
    # 4. Directly calls the underlying ResNet model (model.model if it's a ResNetBaseline)
    #
    # See the ResNetBaseline.forward() method to understand how interpolation currently works.

    # Simple and robust approach for the provided ResNetBaseline wrapper:
    # - If the model exposes `target_size` (see utils/model.py), align it to the
    #   stored `input_size` or the provided `native_size` so that the internal
    #   forward() no longer upsamples to 224x224.
    # - Keep convolutional stem unchanged to avoid weight‑shape mismatches; ResNet
    #   is fully convolutional and supports smaller inputs.
    if hasattr(optimized_model, 'target_size'):
        try:
            # Prefer the model's own recorded input size if available
            desired = getattr(optimized_model, 'input_size', native_size) or native_size
            optimized_model.target_size = int(desired)
        except Exception:
            optimized_model.target_size = int(native_size)
    else:
        # As a generic fallback, attach metadata – downstream code can honor this
        setattr(optimized_model, 'target_size', int(native_size))
        setattr(optimized_model, 'input_size', int(native_size))

    # Report optimization status and provide deployment guidance
    print("INTERPOLATION REMOVAL completed.")
    
    return optimized_model

# --------------------------------------
# DEPTHWISE SEPARABLE CONVOLUTION MODULES
# --------------------------------------

# TODO: Implement this optimization method, if selected in your optimization strategy
def apply_depthwise_separable_optimization(
    model: nn.Module,
    layer_names: Optional[List[str]] = None,
    min_channels: int = 16,
    preserve_residuals: bool = True
) -> nn.Module:
    """
    Convert suitable Conv2d layers to DepthwiseSeparableConv2d for clinical efficiency.
    
    Systematically replaces standard convolutions with depthwise separable alternatives
    to reduce computational cost and memory usage while preserving diagnostic accuracy.
    Essential for deploying medical imaging models on resource-constrained devices.
    
    Args:
        model: Input model to optimize for clinical deployment
        layer_names: Specific layer names to convert (None = convert all suitable layers)
        min_channels: Minimum input/output channels required for conversion
        preserve_residuals: Use residual-compatible configurations for ResNet models
        
    Returns:
        Optimized model with depthwise separable convolutions applied
        
    Note:
        Only converts layers that benefit from depthwise separation (kernel_size > 1,
        sufficient channels, not already grouped). Preserves ResNet compatibility by
        maintaining residual connection requirements.
        
    Example:
        >>> model = create_baseline_model()
        >>> optimized_model = apply_depthwise_separable_optimization(
        ...     model, min_channels=32
        ... )
        >>> # Suitable Conv2d layers now use depthwise separable convolutions
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    replacements = 0  # Track number of successful replacements

    print("Applying depthwise separable convolution optimization...")

    # TODO: Update the model to use depthwise separable convolution instead of convolution. 
    # HINT: To transform a conv2d into depthwise separable, you need to convolve each channel with its own kernel (groups=in_channels) for depthwise, 
    # and then combine information across channels processed by depthwise layer to define the pointwise layer.
    # Note that a conv2d block is also composed by activation and batchnorm in ResNet - Do you want to keep both, either, or none in?
    # Also, think about how the residuals are handled.
    # See https://www.paepper.com/blog/posts/depthwise-separable-convolutions-in-pytorch/ for an intuitive explanation and code template.

    class DepthwisePointwise(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, k: int, s: int, p: int, bias: bool):
            super().__init__()
            self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s,
                                       padding=p, groups=in_ch, bias=False)
            # Keep batchnorm/activation outside to respect original BasicBlock order
            self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1,
                                       padding=0, bias=bias)

        def forward(self, x):
            x = self.depthwise(x)
            x = self.pointwise(x)
            return x

    for name, module in optimized_model.named_modules():
        # Only consider leaf Conv2d modules
        if isinstance(module, nn.Conv2d):
            # Skip pointwise or grouped/depthwise convs; only convert spatial kernels
            k = module.kernel_size[0]
            if k <= 1 or module.groups != 1:
                continue
            if module.in_channels < min_channels or module.out_channels < min_channels:
                continue

            # Respect optional filtering by names
            if layer_names is not None and name not in layer_names:
                continue

            # Prepare replacement
            depthwise_pointwise = DepthwisePointwise(
                in_ch=module.in_channels,
                out_ch=module.out_channels,
                k=module.kernel_size[0],
                s=module.stride[0],
                p=module.padding[0],
                bias=module.bias is not None,
            )

            # Initialize weights with Kaiming norm similar to Conv2d defaults
            for m in depthwise_pointwise.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

            # Replace module in parent
            parent = optimized_model
            subpath = name.split('.')
            for p in subpath[:-1]:
                parent = getattr(parent, p)
            setattr(parent, subpath[-1], depthwise_pointwise)
            replacements += 1

    # Report optimization status
    if replacements > 0:
        print(f"DEPTHWISE SEPARABLE completed: Successfully applied to layers with {replacements} replacements")
    else:
        print("WARNING: DEPTHWISE SEPARABLE not applied: No suitable layers found for replacement")

    return optimized_model

# --------------------------------------
# GROUPED CONVOLUTION MODULES
# --------------------------------------

# TODO: Implement this optimization method, if selected in your optimization strategy
def apply_grouped_convolution_optimization(
    model: nn.Module,
    groups: int = 2,
    min_channels: int = 32,
    layer_names: Optional[List[str]] = None,
    do_depthwise: Optional[bool] = False,
) -> nn.Module:
    """
    Convert suitable Conv2d layers to grouped convolutions for parallel efficiency.
    
    Args:
        model: Input model to optimize
        groups: Number of groups for grouped convolution (typically 2-8)
        min_channels: Minimum channels required for conversion
        layer_names: Specific layers to convert (None = all suitable layers)
        do_depthwise: Whether to apply depthwise grouping (groups=in_channels)
        
    Returns:
        Model with grouped convolutions applied for enhanced efficiency
        
    Note:
        Grouped convolutions can be highly efficient on certain hardware backends, 
        especially when used with memory formats like channels_last and mixed precision (AMP)
        
    Example:
        >>> model = create_baseline_model()
        >>> optimized_model = apply_grouped_convolution_optimization(
        ...     model, groups=4, min_channels=64
        ... )
        >>> # Suitable layers now use 4-group parallel processing
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    # Track number of successful and skipped replacements
    replacements = 0
    skipped = 0

    print(f"Applying grouped convolution optimization (groups={groups})...")

    # TODO: Convert suitable Conv2d layers to grouped convolutions.
    # HINT: Grouped convolution divides input channels into independent groups and applies separate 
    # convolutions to each group. To make this happen, you need to ensure that the later is suitable for this transformation.
    # See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html for how to use the group parameter.

    for name, module in optimized_model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Only convert spatial convs with divisible channels
            if module.kernel_size[0] <= 1 or module.groups != 1:
                continue
            if module.in_channels < min_channels or module.out_channels < min_channels:
                skipped += 1
                continue
            if module.in_channels % groups != 0 or module.out_channels % groups != 0:
                skipped += 1
                continue

            if layer_names is not None and name not in layer_names:
                continue

            new_conv = nn.Conv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.in_channels if do_depthwise else groups,
                bias=(module.bias is not None),
                padding_mode=module.padding_mode,
            )
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            if new_conv.bias is not None:
                nn.init.zeros_(new_conv.bias)

            parent = optimized_model
            subpath = name.split('.')
            for p in subpath[:-1]:
                parent = getattr(parent, p)
            setattr(parent, subpath[-1], new_conv)
            replacements += 1

    # Report optimization status and provide deployment tipes
    if replacements > 0:
        print(f"GROUPED CONV completed: Successfully applied to layers with {replacements} replacements. Skipped {skipped} layers.")
        print("\nDEPLOYMENT TIP: For some hardware (like NVIDIA GPUs), grouped convolutions may require specific memory formats (channels_last) and mixed precision to achieve maximum throughput.")
    else:
        print("WARNING: GROUPED CONV not applied: No suitable layers found for replacement")

    return optimized_model

# --------------------------------------
# INVERTED RESIDUAL BLOCKS
# --------------------------------------

# TODO: Implement this optimization method, if selected in your optimization strategy
def apply_inverted_residual_optimization(
    model: nn.Module,
    target_layers: Optional[List[str]] = None,
    expand_ratio: int = 6
) -> nn.Module:
    """
    Replace suitable blocks with mobile-optimized InvertedResidual blocks.

    Args:
        model: Original model for mobile optimization
        target_layers: Specific layer names to convert (None = auto-detect suitable blocks)
        expand_ratio: Channel expansion factor for inverted residuals (6 is optimal)
        
    Returns:
        Model with mobile-optimized inverted residual blocks
        
    Note:
        This optimization targets BasicBlock structures and converts them to mobile-friendly
        inverted residuals. Most effective for deployment on edge devices and mobile platforms
        common in point-of-care medical applications.
        
    Example:
        >>> model = create_baseline_model()
        >>> mobile_model = apply_inverted_residual_optimization(
        ...     model, expand_ratio=6
        ... )
        >>> # Suitable blocks now use mobile-optimized inverted residuals
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    replacements = 0  # Track number of successful replacements

    print(f"Applying mobile inverted residual optimization...")
    
    # TODO: Replaces suitable blocks in the model with InvertedResidual blocks.
    # HINT: Inverted residuals use an expand→depthwise→project pattern as used in MobileNetV2.
    # The "inverted" aspect means we expand channels first (unlike standard residuals that compress).
    # Architecture flow: input → [expand] → depthwise → project → [+residual]
    #
    # Check the MobileNetV2 code at https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py 
    # for a code template, and consider whether to use ReLU or ReLU6 and batchnorm.

    # Add your code here

    # Report optimization status
    if replacements > 0:
        print(f"INVERTED RESIDUALS completed: Successfully applied to layers with {replacements} replacements")
    else:
        print("WARNING: INVERTED RESIDUALS not applied: No suitable layers found for replacement")

    return optimized_model

# --------------------------------------
# LOW-RANK FACTORIZATION MODULES
# --------------------------------------

# TODO: Implement this optimization method, if selected in your optimization strategy
def apply_lowrank_factorization(
    model: nn.Module,
    min_params: int = 10_000,
    rank_ratio: float = 0.25
) -> nn.Module:
    """
    Apply low-rank factorization to large linear layers for parameter reduction.
    
    Args:
        model: Input model to optimize for clinical deployment
        min_params: Minimum parameter count to consider for factorization
        rank_ratio: Fraction of minimum dimension to use as factorization rank
    
    Returns:
        Model with low-rank factorized linear layers for reduced memory usage
        
    Note:
        Only factorizes layers with sufficient parameters to benefit from compression.
        Rank selection balances compression ratio with accuracy preservation - lower
        ranks provide more compression but may impact diagnostic performance.
        
    Example:
        >>> model = create_baseline_model()
        >>> compressed_model = apply_lowrank_factorization(
        ...     model, min_params=5000, rank_ratio=0.5
        ... )
        >>> # Large linear layers now use low-rank factorization
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    replacements = 0  # Track number of successful replacements

    print("Applying low-rank factorization optimization...")

    # TODO: Factorizes large linear layers into low-rank approximations.
    # HINT: Low-rank factorization decomposes a large weight matrix W into two smaller matrices U and V
    # such that W ≈ U @ V. This dramatically reduces parameters while maintaining representational capacity.
    # Remember that higher rank = better approximation but less compression
    #
    # See https://arikpoz.github.io/posts/2025-04-29-low-rank-factorization-in-pytorch-compressing-neural-networks-with-linear-algebra/ 
    # for explanation and code template, and consider how to initialize parameters with respect to the new rank.

    for name, module in optimized_model.named_modules():
        if isinstance(module, nn.Linear):
            in_f, out_f = module.in_features, module.out_features
            params = in_f * out_f
            if params < min_params:
                continue

            # Determine rank
            r = max(1, int(min(in_f, out_f) * rank_ratio))

            # Build low-rank replacement: Linear(in,r) -> Linear(r,out)
            lr1 = nn.Linear(in_f, r, bias=False)
            lr2 = nn.Linear(r, out_f, bias=True)

            # Initialize using truncated SVD if possible
            try:
                with torch.no_grad():
                    W = module.weight.data
                    # Compute economical SVD on CPU to avoid GPU memory overhead
                    U, S, Vh = torch.linalg.svd(W.cpu(), full_matrices=False)
                    Ur = U[:, :r]
                    Sr = torch.diag(S[:r])
                    Vhr = Vh[:r, :]
                    lr1.weight.data.copy_((Ur @ Sr).to(lr1.weight.data.dtype))
                    lr2.weight.data.copy_(Vhr.to(lr2.weight.data.dtype))
                    if module.bias is not None:
                        lr2.bias.data.copy_(module.bias.data)
                    else:
                        nn.init.zeros_(lr2.bias)
            except Exception:
                # Fallback to Kaiming init
                nn.init.kaiming_uniform_(lr1.weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(lr2.weight, a=math.sqrt(5))
                if lr2.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(lr2.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(lr2.bias, -bound, bound)

            # Replace in parent
            parent = optimized_model
            subpath = name.split('.')
            for p in subpath[:-1]:
                parent = getattr(parent, p)
            setattr(parent, subpath[-1], nn.Sequential(lr1, lr2))
            replacements += 1

    # Report optimization status
    if replacements > 0:
        print(f"LOW RANK FACTORIZATION completed: Successfully applied to layers with {replacements} replacements")
    else:
        print("WARNING: LOW RANK FACTORIZATION not applied: No suitable layers found for replacement")

    return optimized_model

# --------------------------------------
# CHANNEL OPTIMIZATION FUNCTIONS
# --------------------------------------

# TODO: Implement this optimization method, if selected in your optimization strategy
def apply_channel_optimization(
    model: nn.Module,
    enable_channels_last: bool = True,
    enable_inplace_relu: bool = True
) -> nn.Module:
    """
    Apply channel-level optimizations for enhanced hardware efficiency.

    Implements memory layout and activation optimizations to improve hardware utilization
    and reduce memory bandwidth requirements.

    Args:
        model: Model to optimize for hardware efficiency
        enable_channels_last: E.g., you'd use NHWC memory layout for faster GPU convolutions
        enable_inplace_relu: Convert ReLU layers to in-place for memory savings
    
    Returns:
        Hardware-optimized model with improved memory efficiency
        
    Note:
        The 'channels last' memory format can significantly improve convolution performance on certain hardware 
        (e.g., modern GPUs with specialized cores) but requires input tensors to be converted...
        
    Example:
        >>> model = create_baseline_model()
        >>> optimized_model = apply_channel_optimization(model)
        >>> # Remember to convert inputs: input.to(memory_format=torch.channels_last)
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    
    print("Applying channel-level hardware optimizations...")
    
    # TODO: Applies channel-level optimizations such as memory format changes
    # and in-place ReLU conversions for better hardware efficiency.
    # HINT: See https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html for a tutorial on channels last organization,
    # and note how input needs to be handled for it. 
    # Also, consider ensuring activations are in place by reviewing https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948/2 
    # for more details.

    # 1) Mark preferred memory format for downstream tensors
    if enable_channels_last:
        setattr(optimized_model, 'preferred_memory_format', torch.channels_last)

    # 2) Convert ReLU to in-place where safe
    if enable_inplace_relu:
        for name, module in optimized_model.named_modules():
            if isinstance(module, nn.ReLU) and module.inplace is False:
                # Replace with in-place variant
                parent = optimized_model
                subpath = name.split('.')
                for p in subpath[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, subpath[-1], nn.ReLU(inplace=True))

    # Report optimization status
    print("CHANNEL OPTIMIZATION completed")

    return optimized_model

# --------------------------------------
# PARAMETER SHARING FUNCTIONS
# --------------------------------------

# TODO: Implement this optimization method, if selected in your optimization strategy
def apply_parameter_sharing(
    model: nn.Module,
    sharing_groups: Optional[List[List[str]]] = None,
    layer_types: Optional[List[Type[nn.Module]]] = None
) -> nn.Module:
    """
    Apply parameter sharing between layers to reduce memory and improve efficiency.

    Shares weight parameters between layers with identical shapes to reduce memory
    footprint and potentially improve generalization. 

    Args:
        model: Model to optimize through parameter sharing
        sharing_groups: Manual specification of layer groups to share parameters.
                       If None, automatically groups layers with identical weight shapes.
        layer_types: Types of layers to consider for parameter sharing 
                    (defaults to Conv2d for maximum impact)
    
    Returns:
        Memory-optimized model with parameter sharing applied
        
    Note:
        Parameter sharing can improve model generalization by enforcing weight
        consistency across similar layers. Most effective when applied to layers
        with identical computational roles and sufficient parameter count.
        
    Example:
        >>> model = create_baseline_model()
        >>> shared_model = apply_parameter_sharing(model)
        >>> # Layers with identical shapes now share parameters
    """    
    # Default to Conv2d layers (largest parameter count and memory footprint)
    if layer_types is None:
        layer_types = [nn.Conv2d]

    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    # Track number of sharing layers and shared parameters
    total_shared = 0
    total_parameters_shared = 0
    
    print("Applying parameter sharing optimization...")

    # TODO: Shares parameters between layers in specified groups to reduce memory and computation.
    # HINT: Parameter sharing involves assigning the same `nn.Parameter` instance to multiple layers
    #
    # See https://stackoverflow.com/questions/57929299/how-to-share-weights-between-modules-in-pytorch 
    # for some inspiration.

    # Build automatic groups if not provided: layers with identical weight shapes
    from collections import defaultdict
    shape_groups = defaultdict(list)
    if sharing_groups is None:
        for name, module in optimized_model.named_modules():
            if any(isinstance(module, t) for t in layer_types):
                w = getattr(module, 'weight', None)
                if isinstance(w, torch.nn.Parameter):
                    shape_groups[tuple(w.shape)].append((name, module))
        # Keep only groups with more than one layer
        sharing_groups = [[n for n, _ in v] for v in shape_groups.values() if len(v) > 1]

    # Apply sharing within each group
    for group in sharing_groups or []:
        if len(group) < 2:
            continue
        # Use the first layer as the parameter source
        src_name = group[0]
        # Resolve parent and module of the source
        parent = optimized_model
        for p in src_name.split('.')[:-1]:
            parent = getattr(parent, p)
        src_module = getattr(parent, src_name.split('.')[-1])
        src_weight = src_module.weight
        src_bias = getattr(src_module, 'bias', None)

        for tgt_name in group[1:]:
            tgt_parent = optimized_model
            for p in tgt_name.split('.')[:-1]:
                tgt_parent = getattr(tgt_parent, p)
            tgt_module = getattr(tgt_parent, tgt_name.split('.')[-1])
            if tgt_module.weight.shape == src_weight.shape:
                # Point to the same Parameter object
                tgt_module.weight = src_weight
                if hasattr(tgt_module, 'bias') and src_bias is not None and tgt_module.bias is not None:
                    if tgt_module.bias.shape == src_bias.shape:
                        tgt_module.bias = src_bias
                total_shared += 1
                total_parameters_shared += int(src_weight.numel() + (src_bias.numel() if (src_bias is not None and tgt_module.bias is not None) else 0))
   
    # Report optimization status
    if total_shared > 0:
        print(f"PARAMETER SHARING completed - Successfully shared parameters for {total_shared} layers")
        print(f"   Total parameters shared: {total_parameters_shared:,}")
    else:
        print("WARNING: PARAMETER SHARING failed - No suitable layer groups found for optimization")
    
    return optimized_model
