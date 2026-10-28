"""Smoke-test fixtures: a randomly initialised, shrunken copy of each model architecture, with the real processor.

Used to run src/extract.py end to end on a CPU before renting a GPU (catches layout, hook and token-position bugs).

  python tests/make_tiny.py --proc-dir DIR(with the real config + processor files) --out DIR
"""
import argparse
import json
import shutil
from pathlib import Path

import torch


def shrink(cfg):
    if cfg['model_type'] == 'qwen3_5':
        t = cfg['text_config']
        t.update(hidden_size=64, intermediate_size=128, num_hidden_layers=4, head_dim=64, num_attention_heads=2,
                 num_key_value_heads=1, linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
                 linear_num_value_heads=4, mtp_num_hidden_layers=0,
                 layer_types=['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'])
        t['rope_parameters'] = {**t['rope_parameters'], 'mrope_section': [3, 3, 2]}
        cfg['vision_config'].update(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)
    else:
        cfg.update(hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1)
        cfg['rope_scaling'] = {**cfg['rope_scaling'], 'mrope_section': [4, 6, 6]}
        cfg['vision_config'].update(depth=4, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                                    fullatt_block_indexes=[1, 3])
        if 'text_config' in cfg:
            cfg['text_config'].update(hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proc-dir', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from transformers import AutoConfig, AutoModelForImageTextToText
    src, out = Path(a.proc_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(('.safetensors', '.gguf')) and f.name not in ('config.json', 'model.safetensors.index.json'):
            shutil.copy(f, out/f.name)
    cfg = shrink(json.loads((src/'config.json').read_text()))
    (out/'config.json').write_text(json.dumps(cfg))
    config = AutoConfig.from_pretrained(out)
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(config, dtype=torch.float32)
    model.save_pretrained(out)
    print(out.name, sum(p.numel() for p in model.parameters()) / 1e6, 'M params')


if __name__ == '__main__':
    main()
