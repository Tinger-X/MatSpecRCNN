import torch
from torch import nn, Tensor
import torch.nn.functional as F  # noqa
from torchvision.models.detection import _utils as det_utils  # noqa
from torchvision.ops import boxes as box_ops, roi_align, MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor  # noqa
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor  # noqa

from .material import MaterialRoIPool, MaterialHead, MaterialPredictor

__all__ = ["RoIHeads"]


def fast_rcnn_loss(
        class_logits: Tensor,
        box_regression: Tensor,
        labels: list[Tensor],
        regression_targets: list[Tensor]
) -> tuple[Tensor, Tensor]:
    """
    Computes the loss for Faster R-CNN.
    return: (classification_loss, box_loss)
    """
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def maskrcnn_inference(x: Tensor, labels: list[Tensor]) -> list[Tensor]:
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    return: results (list[BoxList]): one BoxList for each image, containing the extra field mask
    """
    mask_prob = x.sigmoid()

    # select masks corresponding to the predicted classes
    num_masks = x.shape[0]
    boxes_per_image = [label.shape[0] for label in labels]
    labels = torch.cat(labels)
    index = torch.arange(num_masks, device=labels.device)
    mask_prob = mask_prob[index, labels][:, None]
    mask_prob = mask_prob.split(boxes_per_image, dim=0)

    return mask_prob


def project_masks_on_boxes(gt_masks: Tensor, boxes: Tensor, matched_idxs: Tensor, size: int) -> Tensor:
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None].to(rois)
    return roi_align(gt_masks, rois, (size, size), 1.0)[:, 0]


def maskrcnn_loss(
        mask_logits: Tensor,
        proposals: list[Tensor],
        gt_masks: list[Tensor],
        gt_labels: list[Tensor],
        mask_matched_idxs: list[Tensor]
) -> Tensor:
    """
    return: mask_loss (Tensor): scalar tensor containing the loss
    """

    discretization_size = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits[torch.arange(labels.shape[0], device=labels.device), labels], mask_targets
    )
    return mask_loss


def expand_boxes(boxes: Tensor, scale: float) -> Tensor:
    w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
    h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
    x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
    y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


def expand_masks(mask: Tensor, padding: int) -> tuple[Tensor, float]:
    m = mask.shape[-1]
    scale = float(m + 2 * padding) / m
    padded_mask = F.pad(mask, (padding,) * 4)
    return padded_mask, scale


def paste_mask_in_image(mask: Tensor, box: Tensor, im_h: int, im_w: int) -> Tensor:
    remove = 1
    w = int(box[2] - box[0] + remove)
    h = int(box[3] - box[1] + remove)
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))
    # Resize mask
    mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
    mask = mask[0][0]

    im_mask = torch.zeros((im_h, im_w), dtype=mask.dtype, device=mask.device)
    # Expected type 'SupportsRichComparisonT ≤: SupportsDunderLT | SupportsDunderGT', got 'Tensor' instead
    x_0 = max(box[0], 0)  # noqa
    x_1 = min(box[2] + 1, im_w)  # noqa
    y_0 = max(box[1], 0)  # noqa
    y_1 = min(box[3] + 1, im_h)  # noqa

    im_mask[y_0:y_1, x_0:x_1] = mask[(y_0 - box[1]): (y_1 - box[1]), (x_0 - box[0]): (x_1 - box[0])]
    return im_mask


def paste_masks_in_image(
        masks: Tensor,
        boxes: Tensor,
        img_shape: tuple[int, int],
        padding: int = 1
) -> Tensor:
    masks, scale = expand_masks(masks, padding=padding)
    boxes = expand_boxes(boxes, scale).to(dtype=torch.int64)
    im_h, im_w = img_shape

    res = [paste_mask_in_image(m[0], b, im_h, im_w) for m, b in zip(masks, boxes)]
    if len(res) > 0:
        ret = torch.stack(res, dim=0)[:, None]
    else:
        ret = masks.new_empty((0, 1, im_h, im_w))
    return ret


class RoIHeads(nn.Module):
    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler": det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(
            self,
            num_classes: int,
            num_materials: int,
            fpn_out_channels: int,
            # Faster R-CNN training
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            # Faster R-CNN inference
            score_thresh=0.5,
            nms_thresh=0.5,
            detections_per_img=100
    ):
        super().__init__()

        self.box_similarity = box_ops.box_iou
        self.proposal_matcher = det_utils.Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        # bbox
        bbox_size = 7
        self.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=bbox_size,
            sampling_ratio=2
        )
        representation_size = 1024
        self.box_head = TwoMLPHead(fpn_out_channels * bbox_size ** 2, representation_size)
        self.box_predictor = FastRCNNPredictor(representation_size, num_classes)

        # mask
        self.mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=14,
            sampling_ratio=2
        )
        mask_layers = [256, 256, 256, 256]
        mask_dilation = 1
        self.mask_head = MaskRCNNHeads(fpn_out_channels, mask_layers, mask_dilation)
        mask_predictor_in_channels = 256
        mask_dim_reduced = 256
        self.mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, num_classes)

        # material
        self.material_roi_pool = MaterialRoIPool(
            featmap_names=["0", "1", "2", "3"],
            output_size=14,
            sampling_ratio=2
        )
        self.material_head = MaterialHead(fpn_out_channels, representation_size)
        self.material_predictor = MaterialPredictor(representation_size, num_materials=num_materials)

    def assign_targets_to_proposals(
            self,
            proposals: list[Tensor],
            gt_boxes: list[Tensor],
            gt_labels: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor]]:
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels: list[Tensor]) -> list[Tensor]:
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    @staticmethod
    def add_gt_proposals(proposals: list[Tensor], gt_boxes: list[Tensor]) -> list[Tensor]:
        proposals = [torch.cat((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]

        return proposals

    def select_training_samples(
            self,
            proposals: list[Tensor],
            targets: list[dict[str, Tensor]],
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"] for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(
            self,
            class_logits: Tensor,
            box_regression: Tensor,
            proposals: list[Tensor],
            image_shapes: list[tuple[int, int]],
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(
            self,
            features: dict[str, Tensor],
            proposals: list[Tensor],
            image_shapes: list[tuple[int, int]],
            targets: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        for t in targets:
            floating_point_types = (torch.float, torch.double, torch.half)
            if not t["boxes"].dtype in floating_point_types:
                raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
            if not t["labels"].dtype == torch.int64:
                raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
            if not t["materials"].dtype == torch.int64:
                raise TypeError(f"target materials must of int64 type, instead got {t['materials'].dtype}")

        # bbox
        proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)
        loss_classifier, loss_box_reg = fast_rcnn_loss(class_logits, box_regression, labels, regression_targets)
        losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}

        # mask
        num_images = len(proposals)
        positive_proposals, positive_indexes = [], []
        for img_id in range(num_images):
            pos = torch.where(labels[img_id] > 0)[0]
            positive_proposals.append(proposals[img_id][pos])
            positive_indexes.append(matched_idxs[img_id][pos])
        mask_features = self.mask_roi_pool(features, positive_proposals, image_shapes)
        mask_features = self.mask_head(mask_features)
        mask_logits = self.mask_predictor(mask_features)
        gt_masks = [t["masks"] for t in targets]
        gt_labels = [t["labels"] for t in targets]
        loss_mask = maskrcnn_loss(mask_logits, positive_proposals, gt_masks, gt_labels, positive_indexes)
        losses.update({"loss_mask": loss_mask})

        # material
        count = sum([p.shape[0] for p in positive_proposals])
        if count == 0:
            return losses
        material_features = self.material_roi_pool(features, positive_proposals, image_shapes)
        material_features = self.material_head(material_features)
        material_logits = self.material_predictor(material_features)
        gt_materials = [t["materials"] for t in targets]
        material_targets = []
        for materials_per_img, midx_per_img in zip(gt_materials, positive_indexes):
            material_targets.append(materials_per_img[midx_per_img])
        material_targets = torch.cat(material_targets, dim=0)
        loss_material = F.cross_entropy(material_logits, material_targets)
        losses.update({"loss_material": loss_material})

        return losses

    def inference(
            self,
            features: dict[str, Tensor],
            proposals: list[Tensor],
            image_shapes: list[tuple[int, int]]
    ):
        result: list[dict[str, torch.Tensor]] = []
        # bbox
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)
        boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append({
                "boxes": boxes[i],
                "labels": labels[i],
                "scores": scores[i],
            })

        # mask
        mask_proposals = boxes  # [p["boxes"] for p in result]
        mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
        mask_features = self.mask_head(mask_features)
        mask_logits = self.mask_predictor(mask_features)
        # labels = [r["labels"] for r in result]
        masks_probs = maskrcnn_inference(mask_logits, labels)
        for mask_prob, r in zip(masks_probs, result):
            full_mask = paste_masks_in_image(mask_prob, r["boxes"], image_shapes[0])
            r["masks"] = full_mask

        # material
        material_proposals = boxes  # [d["boxes"] for d in result]
        count = sum([p.shape[0] for p in material_proposals])
        if count == 0:
            return result
        material_features = self.material_roi_pool(features, material_proposals, image_shapes)
        material_features = self.material_head(material_features)
        material_logits = self.material_predictor(material_features)  # (total_boxes, M)
        material_probs = F.softmax(material_logits, dim=1)
        material_scores, material_preds = material_probs.max(dim=1)
        idx = 0
        for d in result:
            num = d["boxes"].shape[0]
            d["materials"] = material_preds[idx:idx + num]
            d["material_scores"] = material_scores[idx:idx + num]
            idx += num

        return result
