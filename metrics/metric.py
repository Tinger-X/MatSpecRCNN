"""
usage:
```bash
    # run in NVIDIA Tesla T4
    # results saved in `../logs/metrics-result.md`
    export PYTHONPATH=/path/to/MatSpecRCNN
    cd /path/to/MatSpecRCNN/metrics
    # MaskRCNN
    python -u metric.py -m mask-rcnn -b 8
    # MatSpecRCNN-Full
    python -u metric.py -m self-full -b 8
    # MatSpecRCNN-RGB
    python -u metric.py -m self-rgb -b 8
    # MatSpecRCNN-540
    python -u metric.py -m self-540 -b 8
```
"""

import time
import torch
import argparse
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

import utils
import models
from datamaker import Datamaker


class ComprehensiveEvaluator:
    """ 综合评估器，包含多维度指标 """

    def __init__(
            self,
            class_names: list[str],
            material_names: list[str],
            iou_threshold: float = 0.5,
            has_material: bool = True,
    ):
        """
        :param class_names: list[str], 类别名称列表
        :param material_names: list[str], 材质名称列表
        :param iou_threshold: float, IoU匹配阈值
        :param has_material: bool, 是否评估材质预测指标
        """
        self.class_names = class_names
        self.material_names = material_names
        self.iou_threshold = iou_threshold
        self.has_material = has_material

        # 用于检测/分割的统计数据
        self.detection_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
        self.segmentation_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

        # 用于材质分类的统计数据
        self.material_targets = []
        self.material_preds = []

        # 边界框回归指标
        self.bbox_errors = []

        # 推理速度
        self.inference_times = []

    def update(self, targets: list[dict], predictions: list[dict], times: list[float]):
        """ 更新评估器的统计数据

        :param targets: list[dict], 真实标注列表
        :param predictions: list[dict], 预测结果列表
        :param times: list[float], 推理时间列表
        """
        self.inference_times = times
        for idx, (target, pred) in enumerate(zip(targets, predictions)):
            # 计算匹配矩阵
            ious = self._compute_iou_matrix(target["boxes"], pred["boxes"])

            # 1. 目标检测评估
            self._evaluate_detection(target, pred, ious)

            # 2. 实例分割评估
            self._evaluate_segmentation(target, pred, ious)

            # 3. 材质分类评估
            self._evaluate_material(target, pred, ious)

            # 4. 边界框回归评估
            self._evaluate_bbox_regression(target, pred, ious)

    @staticmethod
    def _compute_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor):
        """计算两个边界框集合之间的IoU矩阵"""
        # 左上角点
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        # 右下角点
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

        # 交集区域的宽高
        wh = (rb - lt).clamp(min=0)
        # 交集面积
        inter = wh[:, :, 0] * wh[:, :, 1]

        # 各自面积
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # 并集面积
        union = area1[:, None] + area2 - inter

        # 返回IoU矩阵 [len(boxes1), len(boxes2)]
        return inter / union

    def _evaluate_detection(self, targets, predictions, ious):
        """评估目标检测性能"""
        matched_preds = set()

        # 对每个真实目标寻找最佳匹配
        for t_idx in range(len(targets["boxes"])):
            best_iou = 0
            best_p_idx = -1

            for p_idx in range(len(predictions["boxes"])):
                # 只有类别相同的才考虑匹配
                if targets["labels"][t_idx] != predictions["labels"][p_idx]:
                    continue

                if ious[t_idx, p_idx] > best_iou:
                    best_iou = ious[t_idx, p_idx]
                    best_p_idx = p_idx

            class_id = targets["labels"][t_idx].item()
            class_name = self.class_names[class_id]

            if best_iou >= self.iou_threshold:
                self.detection_stats[class_name]["TP"] += 1
                matched_preds.add(best_p_idx)
            else:
                self.detection_stats[class_name]["FN"] += 1

        # 处理未匹配的预测（误检）
        for p_idx in range(len(predictions["boxes"])):
            if p_idx not in matched_preds:
                class_id = predictions["labels"][p_idx].item()
                class_name = self.class_names[class_id]
                self.detection_stats[class_name]["FP"] += 1

    def _evaluate_segmentation(self, tgt, pred, ious):
        """评估实例分割性能"""
        matched_preds = set()

        # 对每个真实目标寻找最佳匹配
        for t_idx in range(len(tgt["masks"])):
            best_iou = 0
            best_p_idx = -1

            # 计算掩码IoU
            for p_idx in range(len(pred["masks"])):
                # 只有类别相同的才考虑匹配
                if tgt["labels"][t_idx] != pred["labels"][p_idx]:
                    continue

                mask_iou = self._compute_mask_iou(
                    tgt["masks"][t_idx],
                    pred["masks"][p_idx]
                )

                if mask_iou > best_iou:
                    best_iou = mask_iou
                    best_p_idx = p_idx

            class_id = tgt["labels"][t_idx].item()
            class_name = self.class_names[class_id]

            if best_iou >= self.iou_threshold:
                self.segmentation_stats[class_name]["TP"] += 1
                matched_preds.add(best_p_idx)
            else:
                self.segmentation_stats[class_name]["FN"] += 1

        # 处理未匹配的预测（误检）
        for p_idx in range(len(pred["masks"])):
            if p_idx not in matched_preds:
                class_id = pred["labels"][p_idx].item()
                class_name = self.class_names[class_id]
                self.segmentation_stats[class_name]["FP"] += 1

    def _compute_mask_iou(self, mask1, mask2):
        """计算两个掩码之间的IoU"""
        # 确保是二值掩码
        mask1 = (mask1 > 0.5).float()
        mask2 = (mask2 > 0.5).float()

        intersection = torch.sum(mask1 * mask2)
        union = torch.sum(mask1) + torch.sum(mask2) - intersection

        # 避免除以零
        return (intersection / union).item() if union > 0 else 0

    def _evaluate_material(self, tgt, pred, ious):
        """评估材质分类性能"""
        # 对每个真实目标寻找最佳匹配
        if not self.has_material:
            return
        for t_idx in range(len(tgt["materials"])):
            best_iou = 0
            best_p_idx = -1

            for p_idx in range(len(pred["materials"])):
                # 只有类别相同的才考虑匹配
                if tgt["labels"][t_idx] != pred["labels"][p_idx]:
                    continue

                if ious[t_idx, p_idx] > best_iou:
                    best_iou = ious[t_idx, p_idx]
                    best_p_idx = p_idx

            if best_iou >= self.iou_threshold:
                self.material_targets.append(tgt["materials"][t_idx].item())
                self.material_preds.append(pred["materials"][best_p_idx].item())

    def _evaluate_bbox_regression(self, tgt, pred, ious):
        """评估边界框回归性能"""
        # 寻找匹配的真实框和预测框
        matched_pairs = []

        for t_idx in range(len(tgt["boxes"])):
            best_iou = 0
            best_p_idx = -1

            for p_idx in range(len(pred["boxes"])):
                if tgt["labels"][t_idx] != pred["labels"][p_idx]:
                    continue

                if ious[t_idx, p_idx] > best_iou:
                    best_iou = ious[t_idx, p_idx]
                    best_p_idx = p_idx

            if best_iou >= self.iou_threshold:
                matched_pairs.append((t_idx, best_p_idx))

        # 计算匹配框对之间的IoU改进
        for t_idx, p_idx in matched_pairs:
            gt_box = tgt["boxes"][t_idx]
            pred_box = pred["boxes"][p_idx]

            # 计算原始IoU
            original_iou = ious[t_idx, p_idx]

            # 计算IoU改进量
            self.bbox_errors.append(1 - original_iou)

    def record_inference_time(self, batch_time):
        """记录推理时间"""
        self.inference_times.append(batch_time)

    def summarize(self):
        """汇总所有指标"""
        results = {
            # 1. 目标检测指标
            "detection": self._calculate_metrics(self.detection_stats, "Detection"),
            # 2. 实例分割指标
            "segmentation": self._calculate_metrics(self.segmentation_stats, "Segmentation")
        }

        # 3. 材质分类指标
        material = None
        if self.has_material:
            material = {
                "accuracy": np.mean(np.array(self.material_targets) == np.array(self.material_preds)),
                "precision": precision_score(
                    self.material_targets, self.material_preds, average="macro", zero_division=0
                ),
                "recall": recall_score(self.material_targets, self.material_preds, average="macro", zero_division=0),
                "f1": f1_score(self.material_targets, self.material_preds, average="macro", zero_division=0),
                "confusion_matrix": self._confusion_matrix(
                    self.material_targets, self.material_preds, len(self.material_names)
                )
            }
        results["material"] = material

        # 4. 边界框回归指标
        bbox_regression = None
        if len(self.bbox_errors):
            bbox_errors = torch.tensor(self.bbox_errors, device=self.bbox_errors[0].device)
            bbox_regression = {
                "mean_error": bbox_errors.mean().cpu().item(),
                "median_error": bbox_errors.median().cpu().item(),
                "std_error": bbox_errors.std().cpu().item()
            }
        results["bbox_regression"] = bbox_regression

        # 5. 速度指标
        speed = None
        if self.inference_times:
            mean_time = np.mean(self.inference_times)
            speed = {
                "fps": 1 / mean_time if mean_time > 0 else float("inf"),
                "mean_time": mean_time
            }
        results["speed"] = speed

        return results

    def _calculate_metrics(self, stats_dict, metric_type):
        """计算精度、召回率、F1分数和mAP"""
        metrics = {}
        class_metrics = {}
        total_TP, total_FP, total_FN = 0, 0, 0

        for class_name, stats in stats_dict.items():
            TP = stats["TP"]
            FP = stats["FP"]
            FN = stats["FN"]

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            class_metrics[class_name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "TP": TP,
                "FP": FP,
                "FN": FN
            }

            total_TP += TP
            total_FP += FP
            total_FN += FN

        # 全局指标
        global_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        global_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        global_sum = global_precision + global_recall
        global_f1 = 0
        if global_sum > 0:
            global_f1 = 2 * (global_precision * global_recall) / global_sum

        metrics = {
            "global": {
                "precision": global_precision,
                "recall": global_recall,
                "f1": global_f1,
                f"mAP{self.iou_threshold}": self._calculate_mAP(class_metrics)
            },
            "per_class": class_metrics
        }

        return metrics

    @staticmethod
    def _calculate_mAP(class_metrics):
        """计算平均精度(mAP)"""
        aps = []
        for stats in class_metrics.values():
            if stats["TP"] + stats["FP"] > 0:
                aps.append(stats["precision"])
        return np.mean(aps) if aps else 0

    @staticmethod
    def _confusion_matrix(targets, preds, num_classes):
        """生成混淆矩阵"""
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(targets, preds):
            cm[t, p] += 1
        return cm

    def print_report(self, results):
        """打印格式化的评估报告"""
        print("\n===== 综合评估报告 =====")

        # 检测指标
        det = results["detection"]["global"]
        print(f"\n目标检测 (IoU={self.iou_threshold}):")
        print(f"  mAP: {det[f'mAP{self.iou_threshold}']:.4f}")
        print(f"  全局精度: {det['precision']:.4f}")
        print(f"  全局召回率: {det['recall']:.4f}")
        print(f"  全局F1: {det['f1']:.4f}")

        # 分割指标
        seg = results["segmentation"]["global"]
        print(f"\n实例分割 (IoU={self.iou_threshold}):")
        print(f"  mAP: {seg[f'mAP{self.iou_threshold}']:.4f}")
        print(f"  全局精度: {seg['precision']:.4f}")
        print(f"  全局召回率: {seg['recall']:.4f}")
        print(f"  全局F1: {seg['f1']:.4f}")

        # 材质分类
        if results["material"]:
            mat = results["material"]
            print(f"\n材质分类:")
            print(f"  准确率: {mat['accuracy']:.4f}")
            print(f"  宏平均精度: {mat['precision']:.4f}")
            print(f"  宏平均召回率: {mat['recall']:.4f}")
            print(f"  宏平均F1: {mat['f1']:.4f}")

        # 边界框回归
        if results["bbox_regression"]:
            bbox = results["bbox_regression"]
            print(f"\n边界框回归:")
            print(f"  平均IoU误差: {bbox['mean_error']:.4f}")
            print(f"  中值IoU误差: {bbox['median_error']:.4f}")
            print(f"  IoU误差标准差: {bbox['std_error']:.4f}")

        # 速度指标
        if results["speed"]:
            speed = results["speed"]
            print(f"\n推理速度:")
            print(f"  平均推理时间: {speed['mean_time']:.4f}秒/图像")
            print(f"  FPS: {speed['fps']:.2f}")

        print("=========================")


class Timer:
    def __init__(self, batch_size: int):
        self.times = []
        self.batch_size = batch_size

    def exec(self, func: callable, *args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        self.times.append((time.perf_counter() - start_time) / self.batch_size)
        return result


def infer(
        model,
        dataloader: DataLoader,
        batch_size: int,
        has_material: bool
) -> tuple[list[dict], list[dict], list[float]]:
    all_targets, all_predictions = [], []

    timer = Timer(batch_size)
    process = utils.Processor(total=len(dataloader))
    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            predictions = timer.exec(model.inference, images)
            if not has_material:
                for i in range(images.size(0)):
                    del targets[i]["materials"]
            all_targets.extend(targets)
            all_predictions.extend(predictions)
            process.next()

    process.done()
    return all_targets, all_predictions, timer.times


def args_parse():
    parser = argparse.ArgumentParser("Metrics Model")
    parser.add_argument(
        "-m", "--model", type=str, required=True,
        choices=["mask-rcnn", "self-full", "self-rgb", "self-540"], help="Model name"
    )
    parser.add_argument(
        "-b", "--batch", type=int, default=8, help="Training batch size"
    )
    args = parser.parse_args()
    print(
        f"[Arguments] => model: {args.model}, batch: {args.batch}\n"
    )
    return args


def main():
    args = args_parse()
    model_args = (7, 5)
    data_type = "rgb"
    material = True
    if args.model == "mask-rcnn":
        model_args = (7,)
        model_cls = models.MaskRCNN_ResNet50
        material = False
    elif args.model == "self-full":
        model_cls = models.MatSpecRCNN_Full
        data_type = "full"
    elif args.model == "self-rgb":
        model_cls = models.MatSpecRCNN_RGB
    else:
        model_cls = models.MatSpecRCNN_540
        data_type = "540"

    datamaker = Datamaker(
        "../images/bg", "../images/tar", data_type=data_type,  # noqa
        bg_shape=(384, 512), tar_range=(0.4, 0.8)
    )
    model = model_cls(*model_args)
    model.load(f"../weights/{args.model}.pt")
    model.to(utils.GPU)
    val_loader, _ = datamaker.val(batch=args.batch)

    targets, predictions, times = infer(model, val_loader, args.batch, material)
    # 示例类别和材质
    CLASS_NAMES = ["Unknown", "Tank_48-01", "Tank_81", "Tank_T99", "Air_102", "South_215", "Air_Wright"]
    MATERIAL_NAMES = ["Other", "Mental", "Wood", "Plastic", "Paper"]
    # 初始化评估器
    evaluator = ComprehensiveEvaluator(CLASS_NAMES, MATERIAL_NAMES, iou_threshold=0.5, has_material=material)
    # 更新评估器
    evaluator.update(targets, predictions, times)
    # 获取评估结果
    results = evaluator.summarize()
    # 打印报告
    evaluator.print_report(results)


# 使用示例
if __name__ == "__main__":
    main()
