"""Build the per-task T5 table once; training itself does not load T5."""
from __future__ import annotations
import argparse
import os
import tempfile
from pathlib import Path


def save_context(payload, output):
    """Publish a new context file atomically, without replacing an existing one."""
    import torch
    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + output.name + ".", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', required=True, help='The task_strings.json written by data conversion')
    parser.add_argument('--wan-checkpoint-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--tokenizer', help='Local UMT5 tokenizer directory; defaults to the Wan checkpoint copy')
    args = parser.parse_args()
    from rap_wam.task_vocabulary import load_task_strings
    import torch
    from rap_wam.core.model.wan.runtime import T5EncoderModel
    tasks = load_task_strings(args.tasks)
    out = Path(args.out).expanduser()
    if out.exists():
        raise FileExistsError(out)
    base = Path(args.wan_checkpoint_dir).expanduser().resolve()
    tokenizer = args.tokenizer or str(base / 'google/umt5-xxl')
    encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=args.device,
                            checkpoint_path=str(base / 'models_t5_umt5-xxl-enc-bf16.pth'),
                            tokenizer_path=tokenizer)
    table = torch.zeros(len(tasks), 512, 4096, dtype=torch.float32)
    with torch.inference_mode():
        for index, task in enumerate(tasks):
            context = encoder([task], args.device)[0].detach().cpu().float()
            if context.ndim != 2 or context.shape[1] != 4096 or context.shape[0] > 512:
                raise ValueError('Unexpected UMT5 embedding shape')
            if not torch.isfinite(context).all():
                raise ValueError('Nonfinite UMT5 context; no output will be written')
            table[index, :len(context)] = context
    save_context({'tasks': tasks, 'embedding_table': table}, out)
    print(f'Saved context table {tuple(table.shape)} to {out}')


if __name__ == '__main__':
    main()
