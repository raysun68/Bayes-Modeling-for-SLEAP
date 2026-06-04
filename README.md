## Prior.qmd Utilities

My project replaces the arbitrary manual correction of animal pose predictions in SLEAP’s model training with an Extended Kalman Filter (EKF) smoothing correction, so that it improves SLEAP’s missing and unreliable predictions with interpretable results based on continuity in mouse position and the structure of the mouse skeleton. With a 18000-frame 30 FPS video of two mice in a 45.5 x 24 cage as input, the initial SLEAP predictions of Nose, Mid-center and Tail-base are first processed in a tracker which does a forward pass through all frames and corrects body part identities with rules to preserve all reliable frames that display continuity in motion and realistic mouse skeletons. The next step is an EKF smoother based on a random walk state space model, with linearized priors implementing constraints on body-part distances as well as their vector angles. Smoothing is applied on frames that were unreliable based on rules from the previous step, and it greatly improves the accuracy of predictions for these frames. Measuring the prediction accuracy with Root Mean Squared Error (RMSE) on 40 manually labelled random frames, average errors for body parts are reduced to within 1.5 cm, while adequate tuning of parameters can further decrease the RMSE to less than 1 cm, sufficient for behavior analysis. These pose predictions are then imported to SimBA, so that social and freezing behavior bouts are classified with machine learning trained on annotations. Exported behavior bouts are then annotated according to whether the experiment mouse initiates the interaction, and they can be compared with fully manual annotations using a Gantt plot.

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
Full SLEAP+SimBA Animal Social Interaction Labelling Pipeline:

1. Install SLEAP and import experiment video, train centroid model based on 40 manually annotated frames.
2. Convert each new video to 30 FPS, run model on each video to obtain a `.slp` file with predictions for all frames.
3. Convert predictions to `.csv` with `Export.py`, convert manual labels to `.csv` with `Manual.py`.
4. Run `Prior.qmd` to complete pose tracking with realistic skeleton structures and movement.
5. Run `Predictions_1.qmd` to complete Kalman smoothing to fill in body parts for missing frames and reduce MSE of predictions.
6. Convert improved predictions to `.slp` format with `Prediction_import.py`, check for identity swaps and manually correct in a code chunk within `Predictions_1.qmd`.
7. Use `Wide.py` to transform the `.csv` file of improved predictions into SimBA ready format.
8. Import files to SimBA to extract features, train machine learning model for social interactions and obtain bout times.
9. Manually label approach / being approached for each bout to produce final gantt plots of behaviors.


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

