import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet import PointNet


class PointNetPP(nn.Module):
    def __init__(
        self,
        input_size=3,
        output_size=1,
        num_pt_features=[3, 6, 12],
        num_fc_features=[256, 128, 64],
        spatial_dim=3,
        transform_scale=0.001,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_pt_features = num_pt_features
        self.num_fc_features = num_fc_features
        self.spatial_dim = spatial_dim
        self.transform_scale = transform_scale

        self.pointnet_pp_encoder = PointNetPPEncoder(
            input_size=input_size,
            num_pt_features=num_pt_features,
            spatial_dim=spatial_dim,
        )
        self.linear_head = PointNetPPLinearHead(
            input_size=num_pt_features[-1] + spatial_dim,
            output_size=output_size,
            num_fc_features=num_fc_features,
        )

    def forward(self, x):
        x = self.pointnet_pp_encoder(x)
        x = self.linear_head(x)
        return x

    def get_preds_and_transform_matrices(self, x):
        x, matrices = self.pointnet_pp_encoder.get_preds_and_matrices(x)
        x = self.linear_head(x)
        return x, matrices

    def get_transform_loss(self, transforms):
        loss = torch.tensor(0.0).to(transforms[0].device)
        for transform in transforms:
            d = transform.shape[2]
            I = torch.eye(d).to(transform.device)
            loss += torch.mean(
                torch.norm(
                    torch.bmm(transform, transform.transpose(2, 1)) - I,
                    dim=(1, 2),
                )
            )
        return loss


class PointNetPPClassifier(PointNetPP):
    def __init__(self, input_size=3, num_classes=8, **kwargs_pointnetpp):
        kwargs_pointnetpp["input_size"] = input_size
        kwargs_pointnetpp["output_size"] = num_classes
        super().__init__(**kwargs_pointnetpp)

    def forward(self, x):
        logits = super().forward(x)
        return torch.argmax(logits, dim=1)

    def get_loss(self, x, labels):
        preds, transforms = self.get_preds_and_transform_matrices(x)
        transform_loss = self.get_transform_loss(transforms)
        loss = (
            F.cross_entropy(preds.view(x.shape[0], -1), labels.long().view(x.shape[0]))
            + transform_loss * self.transform_scale
        )
        preds = torch.argmax(preds, dim=1).detach()
        return loss, preds


class PointNetPPRegressor(PointNetPP):
    def __init__(
        self, input_size=3, output_size=1, loss_type="l2", **kwargs_pointnetpp
    ):
        accepted_loss_types = ["l1", "l2"]
        assert (
            loss_type in accepted_loss_types
        ), f"loss_type must be in {accepted_loss_types}"
        self.loss_type = loss_type
        kwargs_pointnetpp["input_size"] = input_size
        kwargs_pointnetpp["output_size"] = output_size
        super().__init__(**kwargs_pointnetpp)

    def get_loss(self, x, labels):
        preds, transforms = self.get_preds_and_transform_matrices(x)
        transform_loss = self.get_transform_loss(transforms)
        preds = preds.view(x.shape[0], -1)
        labels = labels.view(x.shape[0], -1).float()
        loss = None
        if self.loss_type == "l1":
            loss = F.l1_loss(preds, labels)
        elif self.loss_type == "l2":
            loss = F.mse_loss(preds, labels)
        return loss + transform_loss * self.transform_scale, preds.detach()


class PointNetPPLinearHead(nn.Module):
    def __init__(self, input_size, output_size, num_fc_features):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_fc_features = num_fc_features

        self.point_net = PointNet(input_size=input_size, output_size=num_fc_features[0])

        self.fcs = nn.Sequential(
            *(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(
                        num_fc_features[i],
                        num_fc_features[i + 1],
                    ),
                )
                for i in range(len(num_fc_features) - 1)
            )
        )

        self.fc_out = nn.Linear(num_fc_features[-1], output_size)

    def forward(self, x):
        x = self.point_net(x)
        x = self.fcs(x)
        x = self.fc_out(x)
        return x


class PointNetPPEncoder(nn.Module):
    """
    PointNet++ encoder module for lidar point cloud data.

    Args:
        input_size (int): Number of input features.
        num_pt_features (list): List of integers representing the number of features
            for each PointNet layer.

    Returns:
        torch.Tensor: Encoded point cloud tensor.
    """

    def __init__(self, input_size, num_pt_features, spatial_dim):
        super().__init__()
        self.input_size = input_size
        self.num_pt_features = num_pt_features
        self.spatial_dim = spatial_dim
        self.point_nets = nn.ModuleList(
            [
                PointNet(
                    input_size=num_pt_features[i - 1] + spatial_dim
                    if i != 0
                    else input_size,
                    output_size=num_pt_features[i],
                )
                for i in range(len(num_pt_features))
            ]
        )

    def forward(self, x):
        return self.get_preds_and_matrices(x)[0]

    def get_preds_and_matrices(self, x):
        # x: (batch_size, num_points, point_dims)
        num_points = x.shape[1]
        num_sample_grouping = len(self.num_pt_features)

        num_centroids = [
            num_points // (2**i) for i in range(1, num_sample_grouping + 1)
        ]

        transform_matrices = [
            torch.zeros(x.shape[0] * num_centroid, 64, 64).to(x.device)
            for num_centroid in num_centroids
        ]

        for i in range(num_sample_grouping):
            x, matrix = self.reduce(x, num_centroids[i], i)
            transform_matrices[i] = matrix

        return x, transform_matrices

    def reduce(self, x, num_centroids, layer_idx=0):
        x, centroids = self.sample_group(x, num_centroids)

        # Applying the PointNet to each group
        x, transform = self.point_nets[layer_idx].get_pred_and_matrix(x)
        x = x.reshape(centroids.shape[0], num_centroids, -1)

        x = torch.cat((x, centroids), dim=2)

        return x, transform

    def sample_group(self, x, num_centroids):
        batch_size = x.shape[0]
        num_points = x.shape[1]
        num_dims = x.shape[2]

        centroids = torch.zeros(batch_size, num_centroids, self.spatial_dim).to(
            x.device
        )
        centroids_idx = torch.zeros(batch_size, num_centroids, dtype=torch.int64).to(
            x.device
        )
        groups = torch.zeros(batch_size, num_points, dtype=torch.int64).to(x.device)

        # Sampling and grouping
        for batch_idx in range(batch_size):
            centroid, centroid_idx = farthest_point_sampling_torch(
                x[batch_idx, :, : self.spatial_dim], num_centroids
            )
            centroids[batch_idx] = centroid.squeeze()
            centroids_idx[batch_idx] = centroid_idx
            groups[batch_idx] = grouping_torch(
                x[batch_idx, :, : self.spatial_dim], centroid
            )

        max_group_length = np.unique(groups.cpu().numpy(), return_counts=True)[1].max()

        grouped_data = torch.zeros(
            batch_size, num_centroids, max_group_length, num_dims
        ).to(x.device)

        # Grouping the points
        for batch_idx in range(batch_size):
            for centroid_idx in range(num_centroids):
                group_length = (groups[batch_idx] == centroid_idx).sum()
                # Group the points in the same centroid
                grouped_data[batch_idx, centroid_idx, :group_length] = x[batch_idx][
                    groups[batch_idx] == centroid_idx
                ]
                # Centralizing the points around the centroid
                grouped_data[
                    batch_idx, centroid_idx, group_length:, : self.spatial_dim
                ] -= centroids[batch_idx, centroid_idx]

        # Making it a tensor of shape (B * # Centroids, N, D)
        x = grouped_data.reshape(batch_size * num_centroids, max_group_length, num_dims)

        return x, centroids


def farthest_point_sampling_numpy(
    points: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """See farthest_point_sampling for documentation."""
    selected_points = np.zeros((k, 1, points.shape[1]))
    selected_points_idx = np.zeros(k, dtype=int)
    start_idx = np.random.randint(0, len(points))
    selected_points[0] = points[start_idx]
    remaining_points = np.delete(points, start_idx, axis=0)
    selected_points_idx[0] = start_idx
    n_selected = 1

    while n_selected < k:
        distances = np.linalg.norm(
            selected_points[:n_selected] - remaining_points, axis=2
        )
        distances_min = np.min(distances, axis=0)
        new_selected_idx = np.argmax(distances_min)

        selected_points[n_selected] = remaining_points[new_selected_idx]
        remaining_points = np.delete(remaining_points, new_selected_idx, axis=0)

        n_selected += 1

    return selected_points, selected_points_idx


def farthest_point_sampling_torch(
    points: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """See farthest_point_sampling for documentation."""
    selected_points = torch.zeros((k, 1, points.shape[1]), device=points.device)
    selected_points_idx = torch.zeros(k, dtype=torch.int64, device=points.device)
    start_idx = torch.randint(0, len(points), (1,))
    selected_points[0] = points[start_idx]
    remaining_points = torch.cat((points[:start_idx], points[start_idx + 1 :]))
    selected_points_idx[0] = start_idx
    n_selected = 1

    while n_selected < k:
        distances = torch.norm(selected_points[:n_selected] - remaining_points, dim=2)
        distances_min = torch.min(distances, dim=0)[0]
        new_selected_idx = torch.argmax(distances_min)

        selected_points[n_selected] = remaining_points[new_selected_idx]
        remaining_points = torch.cat(
            (
                remaining_points[:new_selected_idx],
                remaining_points[new_selected_idx + 1 :],
            )
        )

        n_selected += 1

    return selected_points, selected_points_idx


def farthest_point_sampling(points, k: int):
    """
    Samples k points from the input point cloud using the farthest point sampling algorithm.

    Args:
        points (np.ndarray or torch.Tensor): Input point cloud of shape (n, d) where n is the number of points and d is the dimensionality of each point.
        k (int): Number of points to sample.

    Returns:
        Tuple[np.ndarray, np.ndarray] or Tuple[torch.Tensor, torch.Tensor]: A tuple containing two arrays:
            - selected_points: An array of shape (k, 1, d) containing the k sampled points.
            - selected_points_idx: An array of shape (k,) containing the indices of the k sampled points in the input point cloud.
    """

    if isinstance(points, np.ndarray):
        return farthest_point_sampling_numpy(points, k)
    elif isinstance(points, torch.Tensor):
        return farthest_point_sampling_torch(points, k)
    else:
        raise TypeError("points must be a numpy.ndarray or a torch.Tensor")


def grouping_numpy(points: np.ndarray, selected_points: np.ndarray) -> np.ndarray:
    """See grouping for documentation."""
    distances = np.linalg.norm(points - selected_points, axis=2)
    group_idx = np.argmin(distances, axis=0)

    groups = np.zeros(points.shape[0])
    for i in range(1, selected_points.shape[0]):
        groups[group_idx == i] = i

    return groups


def grouping_torch(points: torch.Tensor, selected_points: torch.Tensor) -> torch.Tensor:
    """See grouping for documentation."""
    distances = torch.norm(points - selected_points, dim=2)
    group_idx = torch.argmin(distances, dim=0)

    groups = torch.zeros(points.shape[0], device=points.device)
    for i in range(1, selected_points.shape[0]):
        groups[group_idx == i] = i

    return groups


def grouping(points, selected_points):
    """
    Group points based on their distance to selected points.

    Args:
        points (numpy.ndarray or torch.Tensor): Array of shape (N, D) representing N points in D-dimensional space.
        selected_points (numpy.ndarray or torch.Tensor): Array of shape (M, D) representing M selected points in D-dimensional space.

    Returns:
        numpy.ndarray or torch.Tensor: Array of shape (N,) representing the group index of each point.
    """

    if isinstance(points, np.ndarray):
        return grouping_numpy(points, selected_points)
    elif isinstance(points, torch.Tensor):
        return grouping_torch(points, selected_points)
    else:
        raise TypeError("points must be a numpy.ndarray or a torch.Tensor")
