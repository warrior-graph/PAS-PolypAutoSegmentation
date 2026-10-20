| Dataset | AP50 | AP50_95 | precision | recall | F1 | best_F1 | mean_best_IoU | fp_per_image |
|---|---|---|---|---|---|---|---|---|
| Kvasir | 0.857 | 0.696 | 0.913 | 0.798 | 0.852 | 0.852 | 0.733 | 0.090 |
| CVC-ClinicDB | 0.991 | 0.803 | 0.970 | 0.941 | 0.955 | 0.956 | 0.850 | 0.032 |
| CVC-ColonDB | 0.834 | 0.525 | 0.918 | 0.708 | 0.799 | 0.807 | 0.604 | 0.063 |
| ETIS-LaribPolypDB | 0.709 | 0.522 | 0.937 | 0.578 | 0.715 | 0.741 | 0.515 | 0.041 |
| CVC-300 | 0.949 | 0.652 | 1.000 | 0.883 | 0.938 | 0.949 | 0.748 | 0.000 |

source=runs/yolo/yolo11n/weights/best.pt tta=False scales=[1.0] imgsz=640 score_thr=0.5
