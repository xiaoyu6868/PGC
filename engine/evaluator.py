import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from utils.metrics import compute_all_metrics, log_metrics


logger = logging.getLogger(__name__)


def evaluate_model(model, device, dataset, batch_size, num_workers):

    model.eval()

    subset_metrics: dict[str, dict[str, float]] = {}
    all_labels: list = []
    all_scores: list = []

    for subset_name in dataset.get_subset_names():
        subset_indices = dataset.get_subset_indices(subset_name)
        subset_dataset = Subset(dataset, subset_indices)
        subset_loader = DataLoader(
            subset_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        subset_labels: list = []
        subset_scores: list = []

        with torch.no_grad():
            for batch in subset_loader:
                # Subset-aware datasets may return either
                #     (image, label, subset_name)   or   (image, label).
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch

                images = images.to(device)
                labels = labels.to(device)

                _features, logits = model(images, return_feature=True)
                logits = logits.view(-1)
                probs = torch.sigmoid(logits)

                subset_labels.extend(labels.cpu().numpy())
                subset_scores.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(probs.cpu().numpy())

        subset_y_true = np.array(subset_labels)
        subset_y_score = np.array(subset_scores)

        metrics = compute_all_metrics(subset_y_true, subset_y_score)
        subset_metrics[subset_name] = metrics
        log_metrics(metrics, subset_name)

    y_true = np.array(all_labels)
    y_score = np.array(all_scores)
    overall_metrics = compute_all_metrics(y_true, y_score)

    return subset_metrics, overall_metrics
