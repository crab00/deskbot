"""YOLOv8n 物体检测（onnxruntime，CPU）。

模型：models/vision/yolov8n.onnx（Ultralytics 导出，输入 1x3x640x640，
输出 1x84x8400：4 个框坐标 + 80 类得分）。检测结果转中文描述，供 LLM 使用。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..utils.logging_setup import get_logger

log = get_logger("vision.detector")

# COCO 80 类（下标→中文名）
COCO_ZH = {
    0: "人", 1: "自行车", 2: "汽车", 3: "摩托车", 4: "飞机", 5: "公交车",
    6: "火车", 7: "卡车", 8: "船", 9: "红绿灯", 10: "消防栓", 11: "停车标志",
    12: "停车计时器", 13: "长椅", 14: "鸟", 15: "猫", 16: "狗", 17: "马",
    18: "羊", 19: "牛", 20: "大象", 21: "熊", 22: "斑马", 23: "长颈鹿",
    24: "背包", 25: "雨伞", 26: "手提包", 27: "领带", 28: "行李箱", 29: "飞盘",
    30: "滑雪板", 31: "滑雪单板", 32: "球", 33: "风筝", 34: "棒球棒",
    35: "棒球手套", 36: "滑板", 37: "冲浪板", 38: "网球拍", 39: "瓶子",
    40: "酒杯", 41: "杯子", 42: "叉子", 43: "刀", 44: "勺子", 45: "碗",
    46: "香蕉", 47: "苹果", 48: "三明治", 49: "橙子", 50: "西兰花",
    51: "胡萝卜", 52: "热狗", 53: "披萨", 54: "甜甜圈", 55: "蛋糕", 56: "椅子",
    57: "沙发", 58: "盆栽", 59: "床", 60: "餐桌", 61: "马桶", 62: "电视",
    63: "笔记本电脑", 64: "鼠标", 65: "遥控器", 66: "键盘", 67: "手机",
    68: "微波炉", 69: "烤箱", 70: "烤面包机", 71: "水槽", 72: "冰箱",
    73: "书", 74: "时钟", 75: "花瓶", 76: "剪刀", 77: "泰迪熊",
    78: "吹风机", 79: "牙刷",
}

INPUT_SIZE = 640
N_CLASSES = 80


def _letterbox(img: np.ndarray, size: int = INPUT_SIZE) -> Tuple[np.ndarray, float, float]:
    """等比缩放并填充到 size×size，返回 (图像, 缩放比, 填充量)。"""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = _resize(img, nw, nh)
    pad_x, pad_y = (size - nw) / 2, (size - nh) / 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[int(pad_y):int(pad_y) + nh, int(pad_x):int(pad_x) + nw] = resized
    return canvas, r, (pad_x, pad_y)


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    try:
        import cv2
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    except Exception:
        return img[:h, :w]


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> List[int]:
    """普通 NMS（按分数降序）。boxes: Nx4 xyxy。"""
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_j - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


class YoloDetector:
    def __init__(self, model_path: Path, confidence: float = 0.35, iou_thr: float = 0.45):
        try:
            import onnxruntime as ort
        except Exception as e:
            raise RuntimeError(f"未安装 onnxruntime: {e}")
        self.confidence = confidence
        self.iou_thr = iou_thr
        self.sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def detect(self, img: np.ndarray) -> List[Dict]:
        """返回 [{label, zh, conf, box(xyxy原图坐标)}]"""
        import cv2  # noqa: F401  # 用 cv2 做 resize 与 letterbox（已有）
        inp, r, (pad_x, pad_y) = _letterbox(img, INPUT_SIZE)
        blob = inp.transpose(2, 0, 1).astype(np.float32)[None] / 255.0
        out = self.sess.run(None, {self.sess.get_inputs()[0].name: blob})[0]
        # out: (1, 84, 8400)
        pred = out[0]  # (84, 8400)
        boxes = pred[:4]          # cx,cy,w,h
        scores = pred[4:]         # (80, 8400)
        cls_ids = scores.argmax(axis=0)
        cls_scores = scores.max(axis=0)
        keep = np.where(cls_scores >= self.confidence)[0]
        results = []
        for idx in keep:
            cx, cy, w, h = boxes[:, idx]
            x1 = (cx - w / 2 - pad_x) / r
            y1 = (cy - h / 2 - pad_y) / r
            x2 = (cx + w / 2 - pad_x) / r
            y2 = (cy + h / 2 - pad_y) / r
            cid = int(cls_ids[idx])
            results.append({
                "label": COCO_ZH.get(cid, f"class{cid}"),
                "conf": float(cls_scores[idx]),
                "box": [x1, y1, x2, y2],
            })
        return results

    def describe(self, img: np.ndarray) -> str:
        """转自然语言描述，如：桌面上有：水杯、键盘、书。"""
        dets = self.detect(img)
        if not dets:
            return "（没有检测到明显的物体）"
        from collections import Counter
        counter = Counter(d["label"] for d in dets)
        parts = []
        for label, cnt in counter.most_common():
            parts.append(label if cnt == 1 else f"{label}×{cnt}")
        return "桌面上有：" + "、".join(parts)
