import torch
import os
from datetime import datetime

ckpt_dir = 'data/output/phase4_checkpoints'
for f in sorted(os.listdir(ckpt_dir)):
    if f.endswith('.pt'):
        fpath = os.path.join(ckpt_dir, f)
        mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
        
        ckpt = torch.load(fpath, map_location='cpu', weights_only=False)
        epoch = ckpt.get('epoch', 'N/A')
        loss = ckpt.get('loss', 'N/A')
        loss_str = f"{loss:.6f}" if isinstance(loss, (int, float)) else str(loss)
        
        print(f'{f:20s} | epoch={epoch:3} | loss={loss_str:10s} | mtime={mtime}')
