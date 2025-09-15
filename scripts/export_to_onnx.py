import argparse
import os
import copy
import torch

from starter.utils.model import create_baseline_model
from starter.utils.architecture_optimization import create_optimized_model


def export_to_onnx(exp_name: str, results_dir: str = "starter/results", fp16: bool = False,
                   dynamic: bool = True, image_size: int = 64):
    results_pkl = os.path.join(results_dir, f"{exp_name}_results.pkl")
    weights_path = os.path.join(results_dir, f"{exp_name}_weights.pth")

    if not os.path.exists(results_pkl):
        raise FileNotFoundError(f"Results file not found: {results_pkl}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    # Recreate optimized model
    # Minimal config assumptions
    from pickle import load
    with open(results_pkl, 'rb') as f:
        results = load(f)

    opt_config = results.get('optimization_config', {
        'interpolation_removal': True,
        'depthwise_separable': True,
        'channel_optimization': False,
        'grouped_conv': False,
        'lowrank_factorization': False,
        'parameter_sharing': False,
    })

    model = create_baseline_model(num_classes=2, input_size=image_size, pretrained=False, fine_tune=False)
    model = create_optimized_model(model, opt_config)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()

    sample = torch.randn(1, 3, image_size, image_size)
    export_model = copy.deepcopy(model)
    if fp16:
        try:
            export_model = export_model.half()
            sample = sample.half()
        except Exception:
            fp16 = False

    os.makedirs(results_dir, exist_ok=True)
    onnx_path = os.path.join(results_dir, f"{exp_name}_deploy.onnx")
    dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch'}} if dynamic else None

    torch.onnx.export(
        export_model,
        sample,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
    )

    try:
        import onnx
        m = onnx.load(onnx_path)
        onnx.checker.check_model(m)
        print(f"OK: exported and validated ONNX at {onnx_path}")
    except Exception as e:
        print(f"Exported but ONNX validation raised a warning: {e}")
        print(f"Model saved at {onnx_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True, help='Experiment name used in Notebook 2 saves')
    ap.add_argument('--fp16', action='store_true', help='Export weights in FP16 where possible')
    ap.add_argument('--no-dynamic', dest='dynamic', action='store_false', help='Disable dynamic batch axis')
    ap.set_defaults(dynamic=True)
    args = ap.parse_args()
    export_to_onnx(args.exp, fp16=args.fp16, dynamic=args.dynamic)

