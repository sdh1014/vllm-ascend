import importlib.util


def main() -> None:
    for module_name in (
        "triton",
        "triton.language",
        "triton.language.extra",
        "triton.language.extra.cann",
        "torch_npu",
    ):
        spec = importlib.util.find_spec(module_name)
        print(module_name, "FOUND" if spec else "MISSING", getattr(spec, "origin", "") if spec else "")

    import torch
    import torch_npu
    from vllm.triton_utils import HAS_TRITON

    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    from vllm_ascend.ops.triton.muls_add import muls_add_triton

    torch.npu.set_device(0)
    print("HAS_TRITON", HAS_TRITON)
    init_device_properties_triton()
    x = torch.ones(1024, dtype=torch.float32, device="npu")
    y = torch.full((1024,), 2.0, dtype=torch.float32, device="npu")
    out = muls_add_triton(x, y, 3.0)
    torch.npu.synchronize()
    print("triton_launch_ok", tuple(out.shape), out.dtype, float(out[0].cpu()))


if __name__ == "__main__":
    main()
