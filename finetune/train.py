import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))

torch.backends.cuda.matmul.allow_tf32 = True

def main():
    from finetune.schemas import Args
    args = Args.parse_args()

    from finetune.models.utils import get_model_cls
    trainer_cls = get_model_cls(args.model_name, args.training_type)

    trainer = trainer_cls(args)
    trainer.fit()


if __name__ == "__main__":
    main()
