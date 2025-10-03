## Prior.qmd Utilities

This repository contains R code (in Prior.qmd) for analyzing and validating multi-animal pose-tracking outputs by estimating prior distributions of body-part distances. The main focus is on building priors for the relative geometry of three key parts: Nose, Mid-center, and Tail-base. The code provides:

Data cleaning and reshaping of pose-tracking outputs and manual annotations.

Computation of pairwise distances (Tail–Nose, Mid–Tail, Mid–Nose) and derived ratios.

Estimation of priors (mean and SD of log distances/ratios) for use in downstream plausibility scoring.

Diagnostic plots (histograms and density overlays) to visualize priors and compare manual vs. automatic distributions.

Unlike earlier versions in Archive_Methods.qmd, this file does not implement anchor-based frame correction or Hungarian assignment tracking, it is focused on prior estimation

---

## Requirements

- R (>= 4.0 recommended)
- R packages:
  - `dplyr`, `tidyr`, `purrr`, `tibble`
  - `clue` (for `solve_LSAP`)
  - `ggplot2`, `plotly` (visualization)
  - (optional) `gganimate` — note: make sure your `ggplot2` version exports `is_ggplot` (some older/newer combos cause `object 'is_ggplot' is not exported` errors)

Install the packages in R:

```r
install.packages(c("dplyr","tidyr","purrr","tibble","clue","ggplot2","plotly"))
# optional:
# install.packages("gganimate")
```
Input Data Format

The code expects pose-tracking data frames with columns:

frame (numeric or integer) — frame index

x_Nose, y_Nose

x_Mid-center, y_Mid-center

x_Tail-base, y_Tail-base

⚠️ Column names with hyphens (e.g. "Mid-center") can be awkward with $.
The QMD uses [[ paste0("x_", bp) ]] accessors, which handle them.
If preferred, rename columns (e.g. "Mid_center") and update body_parts.

A small manual annotation file (predictions_manual.csv) is also used for comparison.

Usage example:

```r
body_parts <- c("Nose", "Mid-center", "Tail-base")
df_corrected_anchor <- correct_instance_frames(
  coords_sort,
  body_parts,
  anchor_bp = "Mid-center",
  instance_col = "instance",
  frame_col = "frame",
  velocity_thresh = 55,
  too_close_thresh = 15
)
```

Main Steps in Prior.qmd

1. Load tracking data and select Nose, Mid-center, and Tail-base coordinates.

2. Compute pairwise distances for each frame:

dtn = Tail–Nose

dtm = Mid–Tail

dnm = Mid–Nose

dist_ratio = dtm / dnm

3. Take log-distances and summarize their means and standard deviations.

4. Define priors:

Priors format:

The function expects priors as the original mapping used in the QMD:

    priors <- list(
      NM = c(mean = pri_dnm_mean, sd = pri_dnm_sd),  # e.g. Nose - Mid-center
      TM = c(mean = pri_dtm_mean, sd = pri_dtm_sd),  # e.g. Tail-base - Mid-center
      TN = c(mean = pri_dtn_mean, sd = pri_dtn_sd)   # e.g. Tail-base - Nose
    )


Usage Example: process_forward_combos

After you’ve run correct_instance_frames() to clean up the raw detections, you can evaluate the per-frame plausibility of the corrected skeletons using priors on pairwise distances.

# Define body parts
```r
body_parts <- c("Nose", "Mid-center", "Tail-base")
```

# Step 1: Correct frame-by-frame assignments (with Mid-center as anchor)
```r
df_corrected <- correct_instance_frames(
  coords_sort,
  body_parts = body_parts,
  anchor_bp = "Mid-center",
  instance_col = "instance",
  frame_col = "frame",
  velocity_thresh = 55,
  too_close_thresh = 15
)
```

# Step 2: Define priors (means / SDs from manual annotations)
```r
priors <- list(
  NM = c(mean = pri_dnm_mean, sd = pri_dnm_sd),  # Nose–Mid-center
  TM = c(mean = pri_dtm_mean, sd = pri_dtm_sd),  # Tail-base–Mid-center
  TN = c(mean = pri_dtn_mean, sd = pri_dtn_sd)   # Tail-base–Nose
)
```

# Step 3: Run forward combos process
```r
res <- process_forward_combos(
  df_corrected,
  r = 1,                         # start frame
  priors = priors,               # priors list
  threshold = 20,                 # score threshold
  anchor_bp = "Mid-center",       # fixed anchor
  other_bps = c("Nose", "Tail-base"),
  motion_lambda = 5,              # motion penalty strength
  motion_alpha = 55,              # velocity decay parameter
  big_penalty = 1e6,              # hard rejection cost
  decay_rate = 0.9,               # decay of priors when unreliable
  too_close_thresh = 15,          # reject if instances overlap too closely
  verbose = TRUE                  # print progress messages
)
```

Plotting & Animation

In addition to prior estimation, Prior.qmd contains snippets to visualize pose-tracking outputs:

Static plots

Histograms of log distances (log_dtn, log_dtm, log_dnm)

Density overlays comparing priors vs. manual annotations

Interactive animations (via Plotly)

Animates trajectories frame-by-frame

Shows Nose (red), Mid-center (blue), Tail-base (green)

Reliable detections: connected markers + lines

Unreliable detections: black dots only

Supports multiple instances (Instance 0 with circles, Instance 1 with triangles)

Legend distinguishes body part + instance (e.g., Nose0, Mid-center1)

Example (simplified):

```r
fig <- plot_ly(df_anim %>% filter(reliable),
               x = ~x, y = ~y, frame = ~frame,
               color = ~body_part_instance,
               type = 'scatter', mode = 'markers+lines') %>%
  add_trace(data = df_anim %>% filter(!reliable),
            x = ~x, y = ~y, frame = ~frame,
            type = 'scatter', mode = 'markers',
            marker = list(color = "black"),
            showlegend = FALSE) %>%
  animation_opts(frame = 40, transition = 0, redraw = FALSE)
```

Performance: The code operates frame-by-frame and loops through combinations; it is not highly vectorized. Reducing frames or pre-filtering can speed up diagnostics. Use verbose = TRUE for periodic messages, or FALSE for faster runs.

## Other files
`Prior.html` is the HTML file containing current results.

`Archive_Methods.qmd` contains code for all methods and data exploration that was previously attempted but discarded due to having flaws or not performing as well as the final method adapted. 

`predictions_aligned_40.csv` contains SLEAP prediction body-part coordinate data from 18071 frames of two mice, with predictions trained from 40 labelled frames.

`predictions_manual.csv` contains manual labeled coordinates for the 40 frames used for training.

