#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml


def deep_merge(base, override):
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def main() -> int:
    parser = argparse.ArgumentParser(description='Render a local light-training config with an algorithm preset.')
    parser.add_argument('--base', required=True, help='Base YAML config path')
    parser.add_argument('--algo', required=True, help='Algorithm preset name')
    parser.add_argument('--out', required=True, help='Output YAML path')
    parser.add_argument('--presets', default='/workspace/ASearcher/ASearcher/configs/light_algo_presets.yaml', help='Preset YAML path')
    parser.add_argument('--list', action='store_true', help='List available presets and exit')
    args = parser.parse_args()

    preset_doc = yaml.safe_load(Path(args.presets).read_text())
    supported = preset_doc.get('supported_light_algorithms', {})
    unsupported = preset_doc.get('unsupported_in_light_trainer', {})

    if args.list:
        print('supported:', ', '.join(sorted(supported)))
        if unsupported:
            print('unsupported:', ', '.join(f'{k} ({v})' for k, v in sorted(unsupported.items())))
        return 0

    if args.algo not in supported:
        print(f'Unknown or unsupported algo: {args.algo}', file=sys.stderr)
        print('Supported:', ', '.join(sorted(supported)), file=sys.stderr)
        if args.algo in unsupported:
            print(f'Unsupported in light trainer: {unsupported[args.algo]}', file=sys.stderr)
        return 2

    base_cfg = yaml.safe_load(Path(args.base).read_text())
    merged = deep_merge(base_cfg, supported[args.algo])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(merged, sort_keys=False))
    print(out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
