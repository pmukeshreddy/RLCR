#!/usr/bin/env bash
# Patches for veRL + SGLang compatibility with FSDP LoRA + layered_summon.
# Run once after: pip install -r requirements.txt
set -e

SGLANG_PATCH="$(python3 -c 'import sglang; print(sglang.__path__[0])')/srt/utils/patch_torch.py"
VERL_PATCH="$(python3 -c 'import verl; print(verl.__path__[0])')/workers/fsdp_workers.py"

echo "Patching SGLang: $SGLANG_PATCH"
python3 -c "
path = '$SGLANG_PATCH'
with open(path) as f: code = f.read()
old = '''    output_args = _modify_tuple(
        output_args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_to_uuid
    )'''
new = '''    if len(output_args) > _REDUCE_TENSOR_ARG_DEVICE_INDEX:
        output_args = _modify_tuple(
            output_args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_to_uuid
        )'''
if old not in code:
    print('  Already patched or version differs — skipping')
else:
    with open(path, 'w') as f: f.write(code.replace(old, new))
    print('  Patched _reduce_tensor_modified: added bounds check for CPU tensors')
"

echo "Patching veRL: $VERL_PATCH"
python3 -c "
path = '$VERL_PATCH'
with open(path) as f: code = f.read()
old = '''        if peft_config is not None and self.base_sync_done:
            per_tensor_param = params.items() if isinstance(params, dict) else params'''
new = '''        if peft_config is not None and self.base_sync_done:
            _device = get_device_id()
            _items = params.items() if isinstance(params, dict) else params
            per_tensor_param = ((name, param.to(_device)) for name, param in _items)'''
if old not in code:
    print('  Already patched or version differs — skipping')
else:
    with open(path, 'w') as f: f.write(code.replace(old, new))
    print('  Patched rollout_mode: move LoRA params to CUDA before IPC serialization')
"

echo "Done. Patches are idempotent — safe to re-run."
