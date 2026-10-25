"""Run verified Step 7 on CPU: 24-image smoke check, or all test images with --full.

Run: /mnt/nas/gezuhao/zhengyuqian/starvla/.venv/bin/python smoke_tsne.py
"""
import argparse
import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def override_assignments(source, replacements):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name in replacements:
                node.value = ast.parse(replacements[name], mode="eval").body
    return compile(ast.fix_missing_locations(tree), "step7_smoke", "exec")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='Run actual Step 7 on the complete test set')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    torch.set_num_threads(4 if args.full else 2)
    notebook = json.loads((root / "image-classifier_verified.ipynb").read_text())
    scope = {}
    with redirect_stdout(io.StringIO()):
        exec(compile("".join(notebook["cells"][4]["source"]), "data_setup", "exec"), scope)
    tree = ast.parse("".join(notebook["cells"][6]["source"]))
    definitions = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "models", "exec"), scope)
    dataset = scope["test_dataset"]
    if args.full:
        # Execute only data/model definitions and Step 7; never run the training cells.
        scope.update(device=torch.device("cpu"), test_loader=torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=False, num_workers=0))
        for index in (14, 15):
            print(f"BEGIN STEP7 CELL {index}", flush=True)
            exec(compile("".join(notebook['cells'][index]['source']), f'step7_cell_{index}', 'exec'), scope)
            print(f"END STEP7 CELL {index}", flush=True)
        return
    # Three real classes with eight images each; use three batches to check concatenation.
    indices = []
    for class_name in ("apple", "bear", "bicycle"):
        matches = [i for i, name in enumerate(dataset.img_names)
                   if Path(name).stem.rsplit("_", 1)[0] == class_name]
        indices.extend(matches[:8])
    assert len(indices) == 24
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=10)
    scope.update(test_loader=loader, device=torch.device("cpu"))
    with tempfile.TemporaryDirectory(prefix="tsne-smoke-", dir=root / ".ipynb_checkpoints") as temporary:
        extraction = "".join(notebook["cells"][14]["source"])
        exec(override_assignments(extraction, {"viz_output_dir": f"Path({temporary!r})"}), scope)
        with np.load(Path(temporary) / "features.npz", allow_pickle=False) as cached:
            expected_labels = [dataset.tags.index(Path(dataset.img_names[i]).stem.rsplit("_", 1)[0]) for i in indices]
            assert np.array_equal(cached["labels"], expected_labels)
            assert cached["sample_paths"].tolist() == [str(Path(dataset.img_names[i]).resolve()) for i in indices]
            assert not np.allclose(cached["rgb_before"], cached["gradient_before"])
            assert not np.allclose(cached["fused_after"], cached["rgb_before"])
            assert json.loads(str(cached["metadata"]))["sample_count"] == len(indices)
        plot = "".join(notebook["cells"][15]["source"])
        settings = dict(n_components=2, perplexity=3, init="pca", learning_rate="auto",
                        max_iter=300, random_state=42, n_jobs=2)
        exec(override_assignments(plot, {"tsne_settings": repr(settings)}), scope)
        with np.load(Path(temporary) / "tsne_embeddings.npz", allow_pickle=False) as cached:
            assert all(cached[name].shape == (24, 2) for name in scope["viz_layers"])
        for name in ("tsne_before_after", "tsne_selected_classes"):
            for extension in ("png", "pdf"):
                assert (Path(temporary) / f"{name}.{extension}").stat().st_size > 1000
        assert not plt.get_fignums()

    # Hook cleanup must also work after an interrupted inference pass.
    model = scope["DualConvNet"](100)
    class InterruptedLoader:
        def __len__(self):
            return 2
        def __iter__(self):
            yield next(iter(loader))
            raise RuntimeError("expected interruption")
    try:
        scope["collect_tsne_features"](model, InterruptedLoader(), torch.device("cpu"))
    except RuntimeError as error:
        assert str(error) == "expected interruption"
    else:
        raise AssertionError("Expected inference interruption")
    assert all(not getattr(model, layer)._forward_hooks for layer, _ in scope["viz_layers"].values())
    assert all(parameter.grad is None for parameter in model.parameters())
    print("PASS: checkpoint loading, three hooks, label alignment, t-SNE, PNG/PDF outputs and hook cleanup")


if __name__ == "__main__":
    main()
