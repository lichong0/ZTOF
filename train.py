import fire
import numpy as np
import torch
import yaml
import json
import os

from helper import (
    accuracy,
    generate_weights,
    load_precomputed_features,
    set_seed,
    load_dataset,
)
from clip import clip
from torchvision.transforms import v2 as T
from torchvision import datasets
from torch.nn import functional as F
from PIL import Image


def main(
    dataset_name: str = "CUB",
    num_workers: int = 8,
    seed: int = 42,
    device: str = "cuda",
):
    device = torch.device(device)
    print("Device:", device)
    print("num_workers:", num_workers)

    with open(file=f"cfgs/{dataset_name}.yaml") as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    set_seed(seed)

    model_size = hparams["model_size"]
    alpha = hparams["alpha"]
    n_samples = hparams["n_samples"]
    batch_size = hparams["batch_size"]
    data_path = hparams["data_path"]

    dataset_for_paths = load_dataset(
        data_path=data_path,
        dataset_name=dataset_name,
        custom_loader=datasets.folder.default_loader
    )
    all_samples = dataset_for_paths.samples
    paths = [p for (p, _) in all_samples]
    class_names = dataset_for_paths.classes

    print(f"Loading {model_size}")
    model, processor = clip.load(model_size, device=device)
    model.eval()
    model.requires_grad_(False)

    def random_crop(image: Image.Image, alpha: float = 0.1) -> Image.Image:
        w, h = image.size
        n_px = np.random.uniform(low=alpha, high=0.9) * min(h, w)
        return T.RandomCrop(int(n_px))(image)

    def custom_loader(path: str) -> torch.Tensor:
        img = datasets.folder.default_loader(path)
        augmented = [processor(img)]
        augmented.extend(processor(random_crop(img)) for _ in range(n_samples))
        return torch.stack(augmented)

    precomputed_features, target, image_features = load_precomputed_features(
        model,
        dataset_name=dataset_name,
        model_size=model_size,
        alpha=alpha,
        n_samples=n_samples,
        batch_size=batch_size,
        num_workers=num_workers,
        data_path=data_path,
        custom_loader=custom_loader,
        device=device,
    )
    max_size = precomputed_features.size(1)
    image_features = image_features.to(device)

    results = {}
    with torch.no_grad():
        methods = hparams["methods"]
        for method in methods:
            method = list(method.values())[0]
            method_name = method["name"]
            method_enabled = method["enabled"]

            if not method_enabled:
                continue

            image_scale = (
                torch.exp(torch.tensor(method["image_scale"])).to(device)
                if "image_scale" in method
                else None
            )

            zeroshot_weights = generate_weights(
                method_name,
                model=model,
                dataset_name=dataset_name,
                tt_scale=text_scale,
                device=device,
            )
            zeroshot_weights = zeroshot_weights.to(image_features.dtype)

            if method_name != "ours":
                logits = image_features.squeeze(1) @ zeroshot_weights
                acc = accuracy(logits, target, image_features.size(0), dataset_name)
                print(f"{method_name}: {acc:.2f}")
                results[method_name] = round(acc, 2)

            else:
                acc_list = []
                patch_num = hparams["patch_n"]
                print(f"n_run: {hparams['n_run']}")

                for run_i in range(hparams["n_run"]):
                    random_indices = torch.randint(0, max_size, (patch_num,))
                    sampled = precomputed_features[:, random_indices, :]
                    patch_embeds = sampled[:, :, :-1]
                    patch_weights = sampled[:, :, -1]
                    del sampled

                    w_i = (patch_weights * image_scale).softmax(-1).unsqueeze(-1)
                    patch_embeds = (patch_embeds * w_i).sum(dim=1)
                    patch_embeds = F.normalize(patch_embeds, dim=-1)

                    logits = patch_embeds @ zeroshot_weights
                    run_acc = accuracy(logits, target, patch_embeds.size(0), dataset_name)
                    acc_list.append(run_acc)

                    if run_i == 0:
                        pred_indices = logits.argmax(dim=1)
                        true_indices = target
                        wrong_indices = torch.nonzero(pred_indices != true_indices).squeeze(1).cpu().tolist()

                        grouped_first = {}
                        for idx in wrong_indices:
                            t_idx = int(true_indices[idx].item())
                            p_idx = int(pred_indices[idx].item())
                            true_name = class_names[t_idx]
                            pred_name = class_names[p_idx]

                            grouped_first.setdefault(true_name, []).append({
                                "path": paths[idx],
                                "pred": pred_name,
                                "true": true_name
                            })

                        os.makedirs("misclassified_cub", exist_ok=True)
                        first_json = os.path.join(
                            "misclassified_cub",
                            f"{dataset_name}_{method_name}_first_run_grouped-ours.json"
                        )
                        with open(first_json, "w", encoding="utf-8") as f:
                            json.dump(grouped_first, f, ensure_ascii=False, indent=4)
                        print(f"  → ：{first_json}")

                mean_acc = np.mean(acc_list)
                std_acc = np.std(acc_list)
                print(f"{method_name}: {mean_acc:.2f}±{std_acc:.2f}")
                results[method_name] = round(mean_acc, 2)

    os.makedirs("misclassified_cub", exist_ok=True)
    summary_path = f"misclassified_cub/{dataset_name}_summary_results-ours.json"
    with open(summary_path, "w", encoding="utf-8") as fout:
        json.dump(results, fout, ensure_ascii=False, indent=4)
    print(f"All done. Summary saved to {summary_path}")

if __name__ == "__main__":
    fire.Fire(main)
