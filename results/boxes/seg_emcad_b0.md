| Dataset | AP50 | AP50_95 | precision | recall | F1 | best_F1 | mean_best_IoU | fp_per_image |
|---|---|---|---|---|---|---|---|---|
| Kvasir | 0.837 | 0.691 | 0.918 | 0.849 | 0.882 | 0.886 | 0.785 | 0.090 |
| CVC-ClinicDB | 0.970 | 0.780 | 0.930 | 0.971 | 0.950 | 0.985 | 0.865 | 0.081 |
| CVC-ColonDB | 0.737 | 0.491 | 0.703 | 0.766 | 0.733 | 0.777 | 0.681 | 0.324 |
| ETIS-LaribPolypDB | 0.704 | 0.508 | 0.629 | 0.748 | 0.683 | 0.780 | 0.687 | 0.464 |
| CVC-300 | 0.924 | 0.640 | 0.862 | 0.933 | 0.896 | 0.926 | 0.815 | 0.150 |

source=/tmp/claude-0/-home-user-PAS-PolypAutoSegmentation/db5b3574-4734-54e9-af26-b34ff0d28298/scratchpad/kres/emcad_b0/best.pth tta=False scales=[1.0] imgsz=640 score_thr=0.5
