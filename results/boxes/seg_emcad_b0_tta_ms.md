| Dataset | AP50 | AP50_95 | precision | recall | F1 | best_F1 | mean_best_IoU | fp_per_image |
|---|---|---|---|---|---|---|---|---|
| Kvasir | 0.825 | 0.679 | 0.884 | 0.832 | 0.857 | 0.868 | 0.785 | 0.130 |
| CVC-ClinicDB | 0.970 | 0.800 | 0.971 | 0.971 | 0.971 | 0.985 | 0.881 | 0.032 |
| CVC-ColonDB | 0.725 | 0.468 | 0.722 | 0.766 | 0.743 | 0.788 | 0.677 | 0.295 |
| ETIS-LaribPolypDB | 0.737 | 0.552 | 0.699 | 0.777 | 0.736 | 0.784 | 0.717 | 0.352 |
| CVC-300 | 0.978 | 0.688 | 0.922 | 0.983 | 0.952 | 0.983 | 0.840 | 0.083 |

source=/tmp/claude-0/-home-user-PAS-PolypAutoSegmentation/db5b3574-4734-54e9-af26-b34ff0d28298/scratchpad/kres/emcad_b0/best.pth tta=True scales=[1.0, 1.25, 1.5] imgsz=640 score_thr=0.5
