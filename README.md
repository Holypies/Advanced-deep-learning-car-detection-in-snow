## Experiment Results

### Main Training Runs

| Run Name | Model | Dataset | imgsz | Freeze | Augmentation | Best mAP50 | mAP50-95 | Precision | Recall | Best Epoch | Stopped |
|----------|-------|---------|-------|--------|--------------|------------|----------|-----------|--------|------------|---------|
| yolo11m_nvd_10p_1280 | YOLOv11m | 10p | 1280 | None | No | 0.683 | 0.374 | 0.742 | 0.601 | 8 | Patience epoch 28 |
| yolov9c_nvd_10p_1280_no_bb | YOLOv9c | 10p | 1280 | None | No | 0.620 | 0.302 | 0.738 | 0.551 | 2 | Patience epoch 21 |
| yolov9c_nvd_10p_1280_frozen_bb | YOLOv9c | 10p | 1280 | 10 | No | 0.679 | 0.335 | 0.771 | 0.600 | 2 | Patience epoch 21 |
| yolov9c_nvd_10p_1280_frozen_aug_v2 | YOLOv9c | 10p | 1280 | 10 | Yes | 0.545 | 0.146 | 0.758 | 0.497 | 8 | Stopped manually epoch 22 |

---

### Augmentation (OAT Sensitivity Analysis)

> Model: YOLOv9c | Dataset: 10p | imgsz: 640 | Epochs: 10 | Freeze: 10
> One parameter varied at a time, others fixed at thesis baseline values
> Source: Parakadavath (2025), LTU MSc Thesis

| Run Name | hsv_s | degrees | scale | Best mAP50 | mAP50-95 | Precision | Recall | Val Box Loss | Best Epoch |
|----------|-------|---------|-------|------------|----------|-----------|--------|--------------|------------|
| baseline | 0.7 | 45.0 | 0.9 | 0.200 | 0.058 | 0.424 | 0.232 | 2.759 | 7 |
| hsvs_low | 0.3 | 45.0 | 0.9 | 0.182 | 0.060 | 0.365 | 0.220 | 2.835 | 3 |
| hsvs_mid | 0.5 | 45.0 | 0.9 | 0.198 | 0.067 | 0.529 | 0.194 | 2.641 | 7 |
| **degrees_low** | **0.7** | **15.0** | **0.9** | **0.280** | **0.103** | **0.481** | **0.304** | **2.576** | **2** |
| degrees_mid | 0.7 | 30.0 | 0.9 | 0.206 | 0.080 | 0.408 | 0.240 | 2.472 | 3 |
| scale_low | 0.7 | 45.0 | 0.5 | 0.209 | 0.058 | 0.501 | 0.229 | 2.827 | 7 |
| scale_mid | 0.7 | 45.0 | 0.7 | 0.184 | 0.054 | 0.395 | 0.225 | 2.970 | 2 |

> **Key finding:** `degrees=15.0` outperforms thesis baseline `degrees=45.0` . hsv_s and scale show minimal sensitivity within tested ranges.
> 
> **Next steps:** I suspect the default lr rate in the model is too high (lr starts from 0.01), potentialy it is worth trying to lower the lr manually (maybe to 0.001) or switch AdamW  to see if the results improve. 
