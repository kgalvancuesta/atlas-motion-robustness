# All-model CV metrics summary

Generated from latest k-fold summary directories only.

| Model | Condition | Clean Dice fold mean +/- std | Aug Dice fold mean +/- std | Robustness delta | Coverage OK | Warnings |
| --- | --- | --- | --- | --- | --- | --- |
| Base CNN | No DA | 0.5188 +/- 0.0264 | 0.4696 +/- 0.0365 | 0.0492 +/- 0.0152 | yes | 0 |
| Base CNN | DA | 0.5185 +/- 0.0198 | 0.4868 +/- 0.0171 | 0.0317 +/- 0.0153 | yes | 0 |
| MedNeXt | No DA | 0.5358 +/- 0.0262 | 0.4923 +/- 0.0257 | 0.0435 +/- 0.0065 | yes | 0 |
| MedNeXt | DA | 0.5432 +/- 0.0234 | 0.5143 +/- 0.0262 | 0.0289 +/- 0.0046 | yes | 0 |
| Swin | No DA | 0.5242 +/- 0.0282 | 0.4759 +/- 0.0214 | 0.0483 +/- 0.0089 | yes | 0 |
| Swin | DA | 0.5388 +/- 0.0223 | 0.5073 +/- 0.0211 | 0.0315 +/- 0.0043 | yes | 0 |
| UXNet | No DA | 0.5279 +/- 0.0338 | 0.4794 +/- 0.0320 | 0.0484 +/- 0.0172 | yes | 0 |
| UXNet | DA | 0.5274 +/- 0.0201 | 0.4953 +/- 0.0197 | 0.0321 +/- 0.0047 | yes | 0 |

## Charts

- `runs/cv_aggregate_summary/all_model_clean_augmented_dice.png`
- `runs/cv_aggregate_summary/all_model_robustness_delta.png`
