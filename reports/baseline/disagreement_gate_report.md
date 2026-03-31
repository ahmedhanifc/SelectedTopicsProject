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
- `enable_disagreement_gate`: `False`
- `disagreement_iou_threshold`: `0.5`
- `disagreement_bad_frames`: `2`
- `enable_boundary_distance_gate`: `False`
- `boundary_distance_threshold`: `20.0`
- `save_disagreement_visuals`: `False`
- `max_disagreement_visuals`: `10`

## Per-Video Summary

| Video | Frames | Segments | Class-change re-prompts | Disagreement re-prompts | Mean IoU | Min IoU | GT Macro IoU | GT Macro Dice | GT Pixel Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| video01 | 958 | 51 | 10 | 0 | 0.8993 | 0.5460 | 0.8455 | 0.8976 | 0.9359 |
| video09 | 240 | 12 | 0 | 0 | 0.9522 | 0.8779 | 0.9210 | 0.9543 | 0.9557 |
| video12 | 486 | 26 | 6 | 0 | 0.8763 | 0.5592 | 0.8328 | 0.8925 | 0.9102 |
| video24 | 890 | 48 | 10 | 0 | 0.8961 | 0.5616 | 0.8733 | 0.9175 | 0.9381 |
| video25 | 80 | 4 | 0 | 0 | 0.9336 | 0.8956 | 0.9316 | 0.9642 | 0.9643 |
| video52 | 808 | 40 | 0 | 0 | 0.9389 | 0.8399 | 0.9359 | 0.9649 | 0.9730 |
| video55 | 241 | 12 | 1 | 0 | 0.8922 | 0.6288 | 0.8548 | 0.9132 | 0.9269 |

## Dataset Summary

| Dataset | Frames | Segments | Class-change re-prompts | Disagreement re-prompts | Mean IoU | Min IoU | GT Macro IoU | GT Macro Dice | GT Pixel Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 3703 | 193 | 27 | 0 | 0.9079 | 0.5460 | 0.8777 | 0.9226 | 0.9424 |

## Interpretation

No disagreement-based re-prompts fired. Class-change re-prompts fired `27` time(s).
 Ground-truth comparison was available for `3533` frame(s): mean GT macro IoU `0.8777`, mean GT macro Dice `0.9226`, mean GT pixel accuracy `0.9424`.

## Sampled IoU Trace

| Video | Frame | Idx | Segment | IoU | FG IoU | GT IoU | GT Dice | GT Acc | Bad | Counter | Class-change | Disagreement | Re-prompt | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- | --- |
| video01 | frame_80_endo | 0 | 1 | 0.9151 | 0.9983 | 0.7852 | 0.8085 | 0.9729 | False | 0 | False | False | False |  |
| video01 | frame_81_endo | 1 | 1 | 0.8692 | 0.9807 | 0.8005 | 0.8165 | 0.9815 | False | 0 | False | False | False |  |
| video01 | frame_82_endo | 2 | 1 | 0.8622 | 0.9822 | 0.7904 | 0.8112 | 0.9787 | False | 0 | False | False | False |  |
| video01 | frame_83_endo | 3 | 1 | 0.8670 | 0.9836 | 0.7958 | 0.8141 | 0.9801 | False | 0 | False | False | False |  |
| video01 | frame_84_endo | 4 | 1 | 0.8787 | 0.9824 | 0.7992 | 0.8159 | 0.9814 | False | 0 | False | False | False |  |
| video01 | frame_85_endo | 5 | 1 | 0.8768 | 0.9815 | 0.7900 | 0.8110 | 0.9789 | False | 0 | False | False | False |  |
| video01 | frame_86_endo | 6 | 1 | 0.8795 | 0.9845 | 0.7972 | 0.8149 | 0.9804 | False | 0 | False | False | False |  |
| video01 | frame_87_endo | 7 | 1 | 0.8831 | 0.9814 | 0.7922 | 0.8122 | 0.9787 | False | 0 | False | False | False |  |
| video01 | frame_88_endo | 8 | 1 | 0.9378 | 0.9870 | 0.7976 | 0.8150 | 0.9757 | False | 0 | False | False | False |  |
| video01 | frame_89_endo | 9 | 1 | 0.9255 | 0.9775 | 0.7951 | 0.8138 | 0.9743 | False | 0 | False | False | False |  |
| video01 | frame_90_endo | 10 | 1 | 0.9156 | 0.9770 | 0.8010 | 0.8168 | 0.9752 | False | 0 | False | False | False |  |
| video01 | frame_91_endo | 11 | 1 | 0.9237 | 0.9811 | 0.7935 | 0.8129 | 0.9722 | False | 0 | False | False | False |  |
| video01 | frame_92_endo | 12 | 1 | 0.8766 | 0.9834 | 0.7932 | 0.8128 | 0.9750 | False | 0 | False | False | False |  |
| video01 | frame_93_endo | 13 | 1 | 0.9102 | 0.9782 | 0.7942 | 0.8133 | 0.9742 | False | 0 | False | False | False |  |
| video01 | frame_94_endo | 14 | 1 | 0.9139 | 0.9829 | 0.7896 | 0.8109 | 0.9724 | False | 0 | False | False | False |  |
| video01 | frame_95_endo | 15 | 1 | 0.9102 | 0.9780 | 0.7871 | 0.8095 | 0.9712 | False | 0 | False | False | False |  |
| video01 | frame_96_endo | 16 | 1 | 0.9099 | 0.9758 | 0.7749 | 0.8029 | 0.9673 | False | 0 | False | False | False |  |
| video01 | frame_97_endo | 17 | 1 | 0.8972 | 0.9729 | 0.7752 | 0.8030 | 0.9688 | False | 0 | False | False | False |  |
| video01 | frame_98_endo | 18 | 1 | 0.9082 | 0.9776 | 0.7741 | 0.8024 | 0.9680 | False | 0 | False | False | False |  |
| video01 | frame_99_endo | 19 | 1 | 0.8945 | 0.9746 | 0.7815 | 0.8064 | 0.9704 | False | 0 | False | False | False |  |
| video01 | frame_100_endo | 20 | 1 | 0.8927 | 0.9746 | 0.7821 | 0.8068 | 0.9697 | False | 0 | False | False | False |  |
| video01 | frame_101_endo | 21 | 1 | 0.8730 | 0.9761 | 0.7723 | 0.8013 | 0.9699 | False | 0 | False | False | False |  |
| video01 | frame_102_endo | 22 | 1 | 0.8803 | 0.9775 | 0.7721 | 0.8011 | 0.9714 | False | 0 | False | False | False |  |
| video01 | frame_103_endo | 23 | 1 | 0.8878 | 0.9721 | 0.7662 | 0.7975 | 0.9707 | False | 0 | False | False | False |  |
| video01 | frame_104_endo | 24 | 1 | 0.8768 | 0.9723 | 0.7684 | 0.7989 | 0.9724 | False | 0 | False | False | False |  |
| video01 | frame_106_endo | 26 | 2 | 0.9616 | 0.9980 | 0.7455 | 0.7854 | 0.9620 | False | 0 | False | False | False |  |
| video01 | frame_107_endo | 27 | 2 | 0.9291 | 0.9830 | 0.7626 | 0.7954 | 0.9683 | False | 0 | False | False | False |  |
| video01 | frame_108_endo | 28 | 2 | 0.8908 | 0.9760 | 0.7635 | 0.7960 | 0.9686 | False | 0 | False | False | False |  |
| video01 | frame_109_endo | 29 | 2 | 0.8811 | 0.9772 | 0.7635 | 0.7961 | 0.9684 | False | 0 | False | False | False |  |
| video01 | frame_110_endo | 30 | 2 | 0.9062 | 0.9772 | 0.7689 | 0.7992 | 0.9694 | False | 0 | False | False | False |  |
| video01 | frame_111_endo | 31 | 2 | 0.8696 | 0.9709 | 0.7709 | 0.8003 | 0.9704 | False | 0 | False | False | False |  |
| video01 | frame_112_endo | 32 | 2 | 0.8906 | 0.9799 | 0.7721 | 0.8009 | 0.9716 | False | 0 | False | False | False |  |
| video01 | frame_113_endo | 33 | 2 | 0.8832 | 0.9753 | 0.7677 | 0.7984 | 0.9713 | False | 0 | False | False | False |  |
| video01 | frame_114_endo | 34 | 2 | 0.8428 | 0.9749 | 0.7670 | 0.7979 | 0.9729 | False | 0 | False | False | False |  |
| video01 | frame_115_endo | 35 | 2 | 0.7298 | 0.9715 | 0.7626 | 0.7952 | 0.9756 | False | 0 | False | False | False |  |
| video01 | frame_116_endo | 36 | 2 | 0.7433 | 0.9742 | 0.7707 | 0.7997 | 0.9793 | False | 0 | False | False | False |  |
| video01 | frame_117_endo | 37 | 2 | 0.7302 | 0.9719 | 0.7720 | 0.8005 | 0.9792 | False | 0 | False | False | False |  |
| video01 | frame_118_endo | 38 | 2 | 0.7340 | 0.9707 | 0.7640 | 0.7957 | 0.9774 | False | 0 | False | False | False |  |
| video01 | frame_119_endo | 39 | 2 | 0.7320 | 0.9730 | 0.8537 | 0.9143 | 0.9781 | False | 0 | False | False | False |  |
| video01 | frame_120_endo | 40 | 2 | 0.7360 | 0.9696 | 0.8522 | 0.9092 | 0.9771 | False | 0 | False | False | False |  |
| video01 | frame_122_endo | 42 | 2 | 0.8568 | 0.9742 | 0.8695 | 0.9261 | 0.9768 | False | 0 | True | False | True | class-change |
| video01 | frame_122_endo | 42 | 3 | 0.9760 | 0.9981 | 0.8532 | 0.9154 | 0.9687 | False | 0 | False | False | False |  |
| video01 | frame_123_endo | 43 | 3 | 0.9201 | 0.9823 | 0.9087 | 0.9510 | 0.9774 | False | 0 | False | False | False |  |
| video01 | frame_125_endo | 45 | 3 | 0.7499 | 0.9725 | 0.8898 | 0.9397 | 0.9746 | False | 0 | True | False | True | class-change |
| video01 | frame_125_endo | 45 | 4 | 0.9626 | 0.9977 | 0.7691 | 0.8098 | 0.9685 | False | 0 | False | False | False |  |
| video01 | frame_126_endo | 46 | 4 | 0.8874 | 0.9756 | 0.7845 | 0.8188 | 0.9728 | False | 0 | False | False | False |  |
| video01 | frame_127_endo | 47 | 4 | 0.8592 | 0.9777 | 0.7699 | 0.8097 | 0.9724 | False | 0 | False | False | False |  |
| video01 | frame_128_endo | 48 | 4 | 0.8505 | 0.9794 | 0.7635 | 0.8052 | 0.9728 | False | 0 | False | False | False |  |
| video01 | frame_129_endo | 49 | 4 | 0.8656 | 0.9757 | 0.7680 | 0.8085 | 0.9737 | False | 0 | False | False | False |  |
| video01 | frame_130_endo | 50 | 4 | 0.9290 | 0.9725 | 0.9081 | 0.9506 | 0.9742 | False | 0 | False | False | False |  |
| video01 | frame_131_endo | 51 | 4 | 0.9355 | 0.9721 | 0.9121 | 0.9532 | 0.9732 | False | 0 | False | False | False |  |
| video01 | frame_132_endo | 52 | 4 | 0.9337 | 0.9724 | 0.9135 | 0.9540 | 0.9715 | False | 0 | False | False | False |  |
| video01 | frame_133_endo | 53 | 4 | 0.9486 | 0.9739 | 0.9195 | 0.9576 | 0.9710 | False | 0 | False | False | False |  |
| video01 | frame_134_endo | 54 | 4 | 0.9413 | 0.9652 | 0.9153 | 0.9553 | 0.9645 | False | 0 | False | False | False |  |
| video01 | frame_135_endo | 55 | 4 | 0.9398 | 0.9634 | 0.9193 | 0.9576 | 0.9640 | False | 0 | False | False | False |  |
| video01 | frame_136_endo | 56 | 4 | 0.9412 | 0.9675 | 0.9151 | 0.9553 | 0.9633 | False | 0 | False | False | False |  |
| video01 | frame_137_endo | 57 | 4 | 0.9370 | 0.9687 | 0.8942 | 0.9428 | 0.9577 | False | 0 | False | False | False |  |
| video01 | frame_138_endo | 58 | 4 | 0.9174 | 0.9567 | 0.8828 | 0.9356 | 0.9545 | False | 0 | False | False | False |  |
| video01 | frame_139_endo | 59 | 4 | 0.9389 | 0.9679 | 0.9193 | 0.9576 | 0.9642 | False | 0 | False | False | False |  |
| video01 | frame_140_endo | 60 | 4 | 0.9120 | 0.9476 | 0.8853 | 0.9366 | 0.9525 | False | 0 | False | False | False |  |
