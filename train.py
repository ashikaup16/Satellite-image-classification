"""Train a four-class DeepLabV3 model from paired RGB images and ID masks."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_resnet50

from config import CLASS_NAMES, MODEL_PATH


class PairedMaskDataset(Dataset):
    """Dataset of same-named images and masks with pixel values 0, 1, 2, or 3."""

    def __init__(self, image_dir: Path, mask_dir: Path, size: int) -> None:
        self.image_paths = sorted(
            path for path in image_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        self.mask_dir = mask_dir
        self.size = size
        if not self.image_paths:
            raise ValueError(f"No training images found in {image_dir}.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        mask_path = self.mask_dir / f"{image_path.stem}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing label mask: {mask_path}")

        image = Image.open(image_path).convert("RGB").resize(
            (self.size, self.size), Image.Resampling.BILINEAR
        )
        mask = Image.open(mask_path).convert("L").resize(
            (self.size, self.size), Image.Resampling.NEAREST
        )
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask, dtype=np.uint8)
        if int(mask_array.max()) >= len(CLASS_NAMES):
            raise ValueError(
                f"{mask_path} must contain only class IDs 0, 1, 2, and 3."
            )
        inputs = torch.from_numpy(image_array).permute(2, 0, 1)
        labels = torch.from_numpy(mask_array.astype(np.int64))
        return inputs, labels


class LogitsOnly(nn.Module):
    """Expose the segmentation logits tensor as the TorchScript model output."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)["out"]


def train(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = PairedMaskDataset(
        args.dataset / "images" / "train",
        args.dataset / "masks" / "train",
        args.image_size,
    )
    validation_set = PairedMaskDataset(
        args.dataset / "images" / "val",
        args.dataset / "masks" / "val",
        args.image_size,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = deeplabv3_resnet50(
        weights=None, weights_backbone=None, num_classes=len(CLASS_NAMES)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        training_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images)["out"], masks)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        model.eval()
        correct_pixels = total_pixels = 0
        with torch.inference_mode():
            for images, masks in validation_loader:
                images, masks = images.to(device), masks.to(device)
                predictions = model(images)["out"].argmax(dim=1)
                correct_pixels += int((predictions == masks).sum().item())
                total_pixels += masks.numel()
        accuracy = correct_pixels / max(total_pixels, 1)
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"loss={training_loss / max(len(train_loader), 1):.4f} "
            f"validation_pixel_accuracy={accuracy:.4f}"
        )

    export_model = LogitsOnly(model.eval()).to(device)
    example = torch.rand(1, 3, args.image_size, args.image_size, device=device)
    traced = torch.jit.trace(export_model, example)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(args.output))
    return args.output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.image_size) <= 0:
        parser.error("epochs, batch-size, and image-size must be positive.")
    return args


if __name__ == "__main__":
    print(f"Saved trained model: {train(parse_args())}")
