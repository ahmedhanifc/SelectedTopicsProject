# Error Analysis Methodology and Technical Discussion

## Scope of the analysis

This error-analysis run evaluates SASVi predictions for the `CHOLECSEG8K` setup stored in `analysis_output`. The report was generated from:

- input frames in `frame_root/video01_28660/`
- predicted masks in `output_masks/video01_28660/`
- ground-truth masks in `gt_root/video01_28660/`
- confidence maps exported during inference in `analysis_output/inference/confidence_maps/video01_28660/`

The analyzed sample contains **80 frames** from **`video01_28660`**.

## Methodology

The workflow has two stages.

### 1. Inference-time export

During SASVi inference, the pipeline is run with `--analysis_output_dir`, which stores per-frame confidence maps and metadata. In this project, that logic is implemented in [src/sam2/eval_sasvi.py](/Users/ahmedhanif/Desktop/dev/SelectedTopics/project/SASVi/src/sam2/eval_sasvi.py). For each frame, the inference stage writes:

- the predicted segmentation mask
- a grayscale confidence map
- per-frame metadata in `analysis_output/inference/inference_metadata.csv`

This stage is useful because it preserves not only the final predicted label map, but also how certain the model was when producing it.

### 2. Offline error analysis

The offline analysis is generated with `analysis_tools/run_error_analysis.py`, which resolves dataset-specific palette configuration from [analysis_tools/config.py](/Users/ahmedhanif/Desktop/dev/SelectedTopics/project/SASVi/analysis_tools/config.py) and computes the full report through [analysis_tools/error_analysis.py](/Users/ahmedhanif/Desktop/dev/SelectedTopics/project/SASVi/analysis_tools/error_analysis.py).

For each predicted frame, the analysis script:

1. Matches the input image, predicted RGB mask, and ground-truth RGB mask by video folder and frame stem.
2. Decodes RGB masks into label-index masks using the dataset palette.
3. Resizes predictions or confidence maps to match the ground-truth resolution when needed.
4. Builds a valid evaluation mask. For `CHOLECSEG8K`, background is label `0`; ignore handling is dataset-configurable.
5. Computes frame-level metrics.
6. Saves visual artifacts for qualitative inspection.

## Technical details of the metrics

For each frame, the following metrics are computed:

- **Pixel accuracy**: fraction of valid pixels whose predicted class matches the ground truth.
- **Macro IoU**: mean IoU across all classes present in either prediction or ground truth for that frame.
- **Macro Dice**: mean Dice score across the same class set.
- **Error pixels / error rate**: total number and proportion of mismatched valid pixels.
- **False positives**: pixels predicted as foreground where the ground truth is background.
- **False negatives**: pixels predicted as background where the ground truth is foreground.
- **Class confusion**: pixels where both prediction and ground truth are foreground, but with different class labels.
- **Temporal IoU to previous prediction**: fraction of valid pixels whose predicted label stayed identical relative to the previous frame.
- **Confidence statistics**: per-frame mean, standard deviation, minimum, and maximum confidence.
- **Per-class IoU and Dice**: class-wise breakdown stored as JSON in `analysis.csv`.

## Visual artifacts

For each frame, the analysis pipeline saves:

- the original image
- colorized prediction
- colorized ground truth
- an error map
- a confidence map
- a combined overlay panel

The error map uses a category-based coloring scheme:

- red: false positives
- blue: false negatives
- yellow: class confusion

When confidence maps are available, confidence is also encoded with diagonal line patterns:

- dense diagonal lines: low confidence (`<= 0.35`)
- medium-spaced diagonal lines: medium confidence (`0.35 < confidence <= 0.60`)
- wide-spaced diagonal lines: high confidence (`> 0.60`)

For the error overlay, the original frame remains visible underneath the visualization as a softened image layer instead of being replaced by a black fill on correct pixels.

This is useful because it separates two different failure modes:

- the model is wrong and uncertain
- the model is wrong or fragile despite appearing temporally stable

## Quantitative results

The report summary in `summary.json` gives the following averages over 80 frames:

- **Mean pixel accuracy:** `0.9294`
- **Mean macro IoU:** `0.7060`
- **Mean macro Dice:** `0.7685`
- **Mean error rate:** `0.0706`

These values indicate that the model is generally strong at the pixel level, with around 93% pixel-wise agreement, while the class-balanced overlap scores are lower. This gap is expected in segmentation because pixel accuracy can remain high even when smaller or harder classes are not segmented well.

### Best and worst frames

The strongest frame in this run is **frame `0029`**:

- pixel accuracy: `0.9418`
- macro IoU: `0.7298`
- macro Dice: `0.7853`
- error rate: `0.0582`

The weakest frame is **frame `0045`**:

- pixel accuracy: `0.8753`
- macro IoU: `0.5671`
- macro Dice: `0.6305`
- error rate: `0.1247`

The most important technical detail about frame `0045` is that its **temporal IoU to the previous prediction is still high (`0.9865`)**. This means the SASVi propagation stayed visually consistent from frame to frame, but it stayed consistent in the wrong direction. In practice, this suggests that temporal stability alone is not enough to guarantee correctness.

### Error composition

Across the full 80-frame run, the error pixels are dominated by:

- **class confusion:** about `49.98%` of all errors
- **false negatives:** about `44.65%`
- **false positives:** about `5.37%`

This means the main failure mode is not spurious foreground generation. Instead, the dominant problems are:

- predicting the wrong foreground class
- missing foreground regions by assigning them to background

That is an important conclusion for model improvement, because it points more toward class disambiguation and object persistence than toward simple background suppression.

### Per-class behavior

Average per-class IoU from the report:

- class `2`: `0.9518`
- class `9`: `0.9190`
- class `6`: `0.8748`
- class `0` (background): `0.8285`
- class `4`: `0.7812`
- class `7`: `0.5946`

The strongest classes are therefore `2`, `9`, and `6`. The weakest recurring class is clearly **class `7`**, and class `4` is also noticeably weaker than the top-performing classes.

There are also two frame-level class-presence anomalies worth noting:

- frame `0030` contains a ground-truth class `10` that is missing from the prediction
- frame `0045` predicts class `9` although class `9` is absent from the ground truth

This explains some of the worst-case drops in macro IoU and shows that rare or transient classes remain difficult.

### Confidence behavior

Confidence is fairly compressed:

- mean confidence ranges from about `0.4064` to `0.4559`
- average confidence decreases from early frames to late frames

By thirds of the video:

- early segment mean macro IoU: `0.7112`, mean confidence: `0.4473`
- middle segment mean macro IoU: `0.7022`, mean confidence: `0.4333`
- late segment mean macro IoU: `0.7048`, mean confidence: `0.4163`

Confidence has only a weak positive relationship with macro IoU, but a clearer negative relationship with error rate. This suggests the confidence map is somewhat useful as a risk signal, although not strong enough yet to be treated as a precise estimator of segmentation quality.

### Temporal behavior

The mean temporal IoU to the previous prediction is **`0.9775`**, which is very high. This confirms that SASVi produces highly stable temporal outputs. However, because the worst frame still has high temporal agreement, this stability should be interpreted carefully:

- it is a strength when the model is already on the correct object track
- it becomes a weakness when an incorrect label assignment persists across successive frames

## Technical interpretation

Overall, the system performs well on dominant regions and common classes, and it maintains very smooth predictions across time. The main technical weaknesses are not random noise. They are more structured:

- foreground regions are sometimes missed entirely
- visually similar or transient classes are confused with one another
- once a wrong interpretation is propagated, the temporal mechanism can preserve that mistake

This means the remaining gap is less about generic denoising and more about better correction, better re-initialization, and stronger class discrimination.

## What can be improved further

### 1. Add explicit failure recovery during propagation

The current results show that high temporal stability can carry errors forward. A practical next step is to add a re-initialization or correction trigger using:

- sudden drops in confidence
- disagreement between current overseer prediction and propagated mask
- large foreground-shape changes between consecutive frames

This would let SASVi recover from drift instead of preserving it.

### 2. Improve class-disambiguation for weaker classes

Since class confusion is the largest error category, the project would benefit from targeted improvements for low-IoU classes, especially class `7` and, secondarily, class `4`. Useful options include:

- class-balanced loss weighting
- targeted augmentation for visually ambiguous classes
- oversampling frames where those classes are present
- adding a confusion-aware post-processing step for pairs of commonly mixed labels

### 3. Improve handling of rare or transient classes

The missing ground-truth class `10` in frame `0030` and the extra predicted class `9` in frame `0045` suggest that rare classes are still fragile. This can be improved by:

- increasing rare-class exposure during training
- adding prompt-refresh logic when new objects appear
- using object-entry and object-exit heuristics instead of relying purely on propagation

### 4. Use confidence more actively, not only for visualization

Right now confidence is exported and visualized, but it is not yet fully used as a control signal. It could be used to:

- trigger overseer refreshes in low-confidence regions
- prioritize human review of high-risk frames
- gate post-processing only where confidence is low

This would make the confidence map operational rather than descriptive.

### 5. Report class-aware temporal failures

The current temporal IoU measures label stability at the whole-frame level. A stronger analysis would also report:

- per-class temporal IoU
- appearance/disappearance events by class
- class-switch events between adjacent frames

That would better expose when a class remains stable for the wrong reason.

### 6. Evaluate on more videos before drawing final conclusions

This run covers only one video (`video01_28660`) with 80 frames. The findings are useful, but they should be treated as a targeted case study rather than a final general claim. The next step should be to repeat the same pipeline across more clips and report:

- per-video mean and variance
- aggregate per-class metrics over the full benchmark
- the consistency of the same failure modes across videos

## Final takeaway

This analysis shows that SASVi already achieves strong pixel-wise segmentation on this clip and is especially strong in temporal consistency. The remaining limitations are concentrated in class confusion, missed foreground regions, and persistent propagation of wrong labels. The most promising improvement direction is therefore not simply “more smoothing,” but a combination of:

- stronger class-specific supervision
- confidence-aware correction
- better recovery when propagated masks drift away from the correct class
