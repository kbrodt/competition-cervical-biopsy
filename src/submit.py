import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
from box import Box

from dataset import DNADSTest
from utils import dict_to_gpu, get_data_groups


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config")
    parser.add_argument(
        "--submit-path", type=str, required=True, help="Path to save submit"
    )

    args = parser.parse_args()

    config = Box.from_yaml(filename=args.config)
    config.test.submit_path = args.submit_path

    return config


def main():
    args = parse_args()
    _ = get_data_groups(args)
    test_anns = pd.read_csv(args.test.test_values)

    models = [
        torch.jit.load(str(p)).cuda().eval()
        for p in Path(args.general.work_dir).rglob("model_last.pt")
    ]
    batch_size = args.test.batch_size
    ds = DNADSTest(test_anns, args)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False,
        collate_fn=DNADSTest.collate_fn,
        pin_memory=True,
    )

    n_models = len(models)
    n_augs = n_models * (args.test.tta + 1)
    print(f"#models: {n_models}, #augs: {n_augs}")
    LOSSES = ["bce", "bfocal"]

    def get_submit():
        logits = torch.zeros(
            (batch_size, args.data.n_classes), dtype=torch.float32, device="cuda"
        )
        logits2 = torch.zeros(
            (batch_size, args.data.n_classes), dtype=torch.float32, device="cuda"
        )
        submit = []
        submit2 = []
        with torch.no_grad():
            with tqdm.tqdm(loader, mininterval=2) as pbar:
                for inp in pbar:
                    anns = inp["label"]
                    del inp["label"]
                    inp = dict_to_gpu(inp)
                    inp["label"] = torch.randn(1)
                    bs = len(inp["feature"])

                    logits.zero_()
                    logits2.zero_()
                    for model in models:
                        logit = model(inp)
                        logits2[:bs] += logit
                        if args.train.loss not in LOSSES:
                            logits[:bs] += torch.softmax(logit, dim=-1)
                        else:
                            logits[:bs] += torch.sigmoid(logit)

                        if args.test.tta > 0:
                            inp2 = {
                                "feature": torch.flip(inp["feature"], dims=[-1]),
                                "label": inp["label"],
                                "aux": inp["aux"],
                            }
                            logit = model(inp2)
                            logits2[:bs] += logit
                            if args.train.loss not in LOSSES:
                                logits[:bs] += torch.softmax(logit, dim=-1)
                            else:
                                logits[:bs] += torch.sigmoid(logit)

                    logits /= n_augs
                    for logit, annotation in zip(logits, anns):
                        logit = logit.cpu().numpy()
                        logit = np.where(
                            logit >= np.partition(logit, -10)[-10], 1.0, 0.0
                        )
                        submit.append([annotation] + logit.tolist())

                    logits2 /= n_augs
                    if args.train.loss not in LOSSES:
                        logits2 = torch.softmax(logits2, dim=-1)
                    else:
                        logits2 = torch.sigmoid(logits2)
                    for logit, annotation in zip(logits2, anns):
                        logit = logit.cpu().numpy()
                        logit = np.where(
                            logit >= np.partition(logit, -10)[-10], 1.0, 0.0
                        )
                        submit2.append([annotation] + logit.tolist())

        return submit, submit2

    submit, submit2 = get_submit()
    submit = pd.DataFrame(submit, columns=["sequence_id"] + args.data.labels)
    submit.to_csv(args.test.submit_path, index=False)

    submit2 = pd.DataFrame(submit2, columns=["sequence_id"] + args.data.labels)
    submit2.to_csv(f"{args.test.submit_path}_2.csv", index=False)


if __name__ == "__main__":
    main()
