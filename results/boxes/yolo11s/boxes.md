| Dataset | AP50 | AP50_95 | precision | recall | F1 | best_F1 | mean_best_IoU | fp_per_image |
|---|---|---|---|---|---|---|---|---|
| Kvasir | 0.879 | 0.708 | 0.904 | 0.790 | 0.843 | 0.851 | 0.726 | 0.100 |
| CVC-ClinicDB | 0.991 | 0.785 | 0.956 | 0.956 | 0.956 | 0.963 | 0.853 | 0.048 |
| CVC-ColonDB | 0.812 | 0.529 | 0.891 | 0.647 | 0.750 | 0.761 | 0.562 | 0.079 |
| ETIS-LaribPolypDB | 0.716 | 0.532 | 0.899 | 0.563 | 0.693 | 0.724 | 0.511 | 0.066 |
| CVC-300 | 0.924 | 0.648 | 0.981 | 0.867 | 0.920 | 0.920 | 0.750 | 0.017 |

source=runs/yolo/yolo11s/weights/best.pt tta=False scales=[1.0] imgsz=640 score_thr=0.5
