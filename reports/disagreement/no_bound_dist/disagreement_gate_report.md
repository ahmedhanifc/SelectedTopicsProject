# Disagreement-Gated Re-prompting Report

## Configuration

- `device`: `cuda`
- `dataset_type`: `CHOLECSEG8K`
- `base_video_dir`: `C:\Users\Test\Desktop\SelectedTopicsProject\dataset`
- `gt_root_dir`: `None`
- `video_name`: `None`
- `video_names`: `None`
- `frame_name`: `None`
- `start_frame`: `None`
- `end_frame`: `None`
- `enable_disagreement_gate`: `True`
- `disagreement_iou_threshold`: `0.95`
- `disagreement_bad_frames`: `2`
- `enable_boundary_distance_gate`: `False`
- `boundary_distance_threshold`: `20.0`
- `save_disagreement_visuals`: `True`
- `max_disagreement_visuals`: `6`

## Per-Video Summary

| Video | Frames | Segments | Class-change re-prompts | Disagreement re-prompts | Mean IoU | Min IoU | GT Macro IoU | GT Macro Dice | GT Pixel Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| video01 | 1319 | 383 | 10 | 361 | 0.9323 | 0.6698 | 0.8502 | 0.9021 | 0.9383 |
| video09 | 253 | 20 | 0 | 13 | 0.9612 | 0.9109 | 0.9261 | 0.9575 | 0.9594 |
| video12 | 714 | 240 | 5 | 229 | 0.9170 | 0.5995 | 0.8250 | 0.8880 | 0.8985 |
| video24 | 1240 | 371 | 7 | 353 | 0.9333 | 0.4088 | 0.8532 | 0.9086 | 0.9149 |
| video25 | 107 | 28 | 0 | 27 | 0.9554 | 0.9334 | 0.9210 | 0.9584 | 0.9549 |
| video52 | 1005 | 210 | 0 | 197 | 0.9515 | 0.8440 | 0.9295 | 0.9612 | 0.9691 |
| video55 | 336 | 99 | 1 | 95 | 0.9379 | 0.7137 | 0.8569 | 0.9132 | 0.9238 |

## Dataset Summary

| Dataset | Frames | Segments | Class-change re-prompts | Disagreement re-prompts | Mean IoU | Min IoU | GT Macro IoU | GT Macro Dice | GT Pixel Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 4974 | 1351 | 23 | 1275 | 0.9366 | 0.4088 | 0.8692 | 0.9184 | 0.9333 |

## Interpretation

Disagreement gating triggered `1275` corrective re-prompt(s), in addition to `23` class-change re-prompt(s).
 Ground-truth comparison was available for `4905` frame(s): mean GT macro IoU `0.8692`, mean GT macro Dice `0.9184`, mean GT pixel accuracy `0.9333`.

## Sampled IoU Trace

| Video | Frame | Idx | Segment | IoU | FG IoU | GT IoU | GT Dice | GT Acc | Bad | Counter | Class-change | Disagreement | Re-prompt | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- | --- |
| video01 | frame_80_endo | 0 | 1 | 0.9151 | 0.9983 | 0.7852 | 0.8085 | 0.9729 | False | 0 | False | False | False |  |
| video01 | frame_81_endo | 1 | 1 | 0.8692 | 0.9807 | 0.8005 | 0.8165 | 0.9815 | True | 1 | False | False | False |  |
| video01 | frame_82_endo | 2 | 1 | 0.8622 | 0.9822 | 0.7904 | 0.8112 | 0.9787 | True | 2 | False | True | True | disagreement |
| video01 | frame_82_endo | 2 | 2 | 0.9007 | 0.9983 | 0.7764 | 0.8037 | 0.9710 | False | 0 | False | False | False |  |
| video01 | frame_83_endo | 3 | 2 | 0.8902 | 0.9827 | 0.7892 | 0.8106 | 0.9782 | True | 1 | False | False | False |  |
| video01 | frame_84_endo | 4 | 2 | 0.8894 | 0.9794 | 0.7929 | 0.8125 | 0.9796 | True | 2 | False | True | True | disagreement |
| video01 | frame_84_endo | 4 | 3 | 0.8969 | 0.9984 | 0.7806 | 0.8060 | 0.9695 | False | 0 | False | False | False |  |
| video01 | frame_85_endo | 5 | 3 | 0.8717 | 0.9809 | 0.7778 | 0.8045 | 0.9723 | True | 1 | False | False | False |  |
| video01 | frame_86_endo | 6 | 3 | 0.8752 | 0.9820 | 0.7879 | 0.8099 | 0.9745 | True | 2 | False | True | True | disagreement |
| video01 | frame_86_endo | 6 | 4 | 0.9302 | 0.9985 | 0.7849 | 0.8083 | 0.9735 | False | 0 | False | False | False |  |
| video01 | frame_87_endo | 7 | 4 | 0.8924 | 0.9795 | 0.7905 | 0.8112 | 0.9790 | True | 1 | False | False | False |  |
| video01 | frame_88_endo | 8 | 4 | 0.8703 | 0.9869 | 0.7956 | 0.8140 | 0.9793 | True | 2 | False | True | True | disagreement |
| video01 | frame_88_endo | 8 | 5 | 0.9365 | 0.9986 | 0.7884 | 0.8102 | 0.9730 | False | 0 | False | False | False |  |
| video01 | frame_89_endo | 9 | 5 | 0.9264 | 0.9747 | 0.7930 | 0.8126 | 0.9753 | True | 1 | False | False | False |  |
| video01 | frame_90_endo | 10 | 5 | 0.9187 | 0.9733 | 0.7966 | 0.8145 | 0.9752 | True | 2 | False | True | True | disagreement |
| video01 | frame_90_endo | 10 | 6 | 0.9511 | 0.9982 | 0.7795 | 0.8053 | 0.9608 | False | 0 | False | False | False |  |
| video01 | frame_91_endo | 11 | 6 | 0.9403 | 0.9759 | 0.7828 | 0.8071 | 0.9623 | True | 1 | False | False | False |  |
| video01 | frame_92_endo | 12 | 6 | 0.9451 | 0.9813 | 0.7828 | 0.8071 | 0.9630 | True | 2 | False | True | True | disagreement |
| video01 | frame_92_endo | 12 | 7 | 0.9644 | 0.9983 | 0.7796 | 0.8055 | 0.9643 | False | 0 | False | False | False |  |
| video01 | frame_93_endo | 13 | 7 | 0.9407 | 0.9807 | 0.7856 | 0.8087 | 0.9671 | True | 1 | False | False | False |  |
| video01 | frame_94_endo | 14 | 7 | 0.9446 | 0.9843 | 0.7838 | 0.8077 | 0.9665 | True | 2 | False | True | True | disagreement |
| video01 | frame_94_endo | 14 | 8 | 0.9581 | 0.9983 | 0.7806 | 0.8061 | 0.9650 | False | 0 | False | False | False |  |
| video01 | frame_95_endo | 15 | 8 | 0.9408 | 0.9837 | 0.7851 | 0.8085 | 0.9683 | True | 1 | False | False | False |  |
| video01 | frame_96_endo | 16 | 8 | 0.9373 | 0.9812 | 0.7714 | 0.8010 | 0.9642 | True | 2 | False | True | True | disagreement |
| video01 | frame_96_endo | 16 | 9 | 0.9665 | 0.9983 | 0.7723 | 0.8016 | 0.9622 | False | 0 | False | False | False |  |
| video01 | frame_97_endo | 17 | 9 | 0.9390 | 0.9790 | 0.7810 | 0.8062 | 0.9651 | True | 1 | False | False | False |  |
| video01 | frame_98_endo | 18 | 9 | 0.9365 | 0.9793 | 0.7804 | 0.8059 | 0.9642 | True | 2 | False | True | True | disagreement |
| video01 | frame_98_endo | 18 | 10 | 0.9768 | 0.9981 | 0.7708 | 0.8007 | 0.9593 | False | 0 | False | False | False |  |
| video01 | frame_99_endo | 19 | 10 | 0.9547 | 0.9841 | 0.7806 | 0.8060 | 0.9627 | False | 0 | False | False | False |  |
| video01 | frame_100_endo | 20 | 10 | 0.9546 | 0.9837 | 0.7788 | 0.8050 | 0.9628 | False | 0 | False | False | False |  |
| video01 | frame_101_endo | 21 | 10 | 0.9288 | 0.9824 | 0.7742 | 0.8025 | 0.9631 | True | 1 | False | False | False |  |
| video01 | frame_102_endo | 22 | 10 | 0.9360 | 0.9830 | 0.7767 | 0.8039 | 0.9639 | True | 2 | False | True | True | disagreement |
| video01 | frame_102_endo | 22 | 11 | 0.9746 | 0.9980 | 0.7643 | 0.7969 | 0.9613 | False | 0 | False | False | False |  |
| video01 | frame_103_endo | 23 | 11 | 0.9299 | 0.9745 | 0.7663 | 0.7975 | 0.9667 | True | 1 | False | False | False |  |
| video01 | frame_104_endo | 24 | 11 | 0.9222 | 0.9767 | 0.7601 | 0.7939 | 0.9668 | True | 2 | False | True | True | disagreement |
| video01 | frame_104_endo | 24 | 12 | 0.9660 | 0.9979 | 0.7462 | 0.7860 | 0.9601 | False | 0 | False | False | False |  |
| video01 | frame_105_endo | 25 | 12 | 0.9396 | 0.9816 | 0.7666 | 0.7980 | 0.9664 | True | 1 | False | False | False |  |
| video01 | frame_106_endo | 26 | 12 | 0.9375 | 0.9819 | 0.7552 | 0.7911 | 0.9659 | True | 2 | False | True | True | disagreement |
| video01 | frame_106_endo | 26 | 13 | 0.9616 | 0.9980 | 0.7455 | 0.7854 | 0.9620 | False | 0 | False | False | False |  |
| video01 | frame_107_endo | 27 | 13 | 0.9291 | 0.9830 | 0.7626 | 0.7954 | 0.9683 | True | 1 | False | False | False |  |
| video01 | frame_108_endo | 28 | 13 | 0.8908 | 0.9760 | 0.7635 | 0.7960 | 0.9686 | True | 2 | False | True | True | disagreement |
| video01 | frame_108_endo | 28 | 14 | 0.9291 | 0.9980 | 0.7750 | 0.8029 | 0.9672 | False | 0 | False | False | False |  |
| video01 | frame_109_endo | 29 | 14 | 0.8960 | 0.9823 | 0.7680 | 0.7988 | 0.9704 | True | 1 | False | False | False |  |
| video01 | frame_110_endo | 30 | 14 | 0.8882 | 0.9773 | 0.7754 | 0.8030 | 0.9718 | True | 2 | False | True | True | disagreement |
| video01 | frame_110_endo | 30 | 15 | 0.9532 | 0.9981 | 0.7740 | 0.8024 | 0.9634 | False | 0 | False | False | False |  |
| video01 | frame_111_endo | 31 | 15 | 0.8947 | 0.9731 | 0.7833 | 0.8074 | 0.9703 | True | 1 | False | False | False |  |
| video01 | frame_112_endo | 32 | 15 | 0.8927 | 0.9791 | 0.7802 | 0.8057 | 0.9716 | True | 2 | False | True | True | disagreement |
| video01 | frame_112_endo | 32 | 16 | 0.9429 | 0.9982 | 0.7675 | 0.7985 | 0.9698 | False | 0 | False | False | False |  |
| video01 | frame_113_endo | 33 | 16 | 0.8961 | 0.9791 | 0.7721 | 0.8009 | 0.9747 | True | 1 | False | False | False |  |
| video01 | frame_114_endo | 34 | 16 | 0.8920 | 0.9800 | 0.7723 | 0.8010 | 0.9764 | True | 2 | False | True | True | disagreement |
| video01 | frame_114_endo | 34 | 17 | 0.9277 | 0.9983 | 0.7735 | 0.8020 | 0.9710 | False | 0 | False | False | False |  |
| video01 | frame_115_endo | 35 | 17 | 0.7463 | 0.9771 | 0.7792 | 0.8050 | 0.9770 | True | 1 | False | False | False |  |
| video01 | frame_116_endo | 36 | 17 | 0.7600 | 0.9809 | 0.7771 | 0.8037 | 0.9786 | True | 2 | False | True | True | disagreement |
| video01 | frame_116_endo | 36 | 18 | 0.9737 | 0.9985 | 0.9249 | 0.9604 | 0.9733 | False | 0 | False | False | False |  |
| video01 | frame_117_endo | 37 | 18 | 0.9259 | 0.9780 | 0.9233 | 0.9593 | 0.9752 | True | 1 | False | False | False |  |
| video01 | frame_118_endo | 38 | 18 | 0.9272 | 0.9748 | 0.9083 | 0.9500 | 0.9758 | True | 2 | False | True | True | disagreement |
| video01 | frame_118_endo | 38 | 19 | 0.9858 | 0.9982 | 0.9137 | 0.9540 | 0.9707 | False | 0 | False | False | False |  |
| video01 | frame_119_endo | 39 | 19 | 0.9245 | 0.9824 | 0.7475 | 0.7858 | 0.9729 | True | 1 | False | False | False |  |
| video01 | frame_120_endo | 40 | 19 | 0.9374 | 0.9794 | 0.7754 | 0.8028 | 0.9734 | True | 2 | False | True | True | disagreement |
| video01 | frame_120_endo | 40 | 20 | 0.9867 | 0.9981 | 0.7639 | 0.7966 | 0.9661 | False | 0 | False | False | False |  |
