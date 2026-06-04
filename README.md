## Prior.qmd Utilities

My project replaces the arbitrary manual correction of animal pose predictions in SLEAP’s model training with an Extended Kalman Filter (EKF) smoothing correction, so that it improves SLEAP’s missing and unreliable predictions with interpretable results based on continuity in mouse position and the structure of the mouse skeleton. With a 18000-frame 30 FPS video of two mice in a 45.5 x 24 cage as input, the initial SLEAP predictions of Nose, Mid-center and Tail-base are first processed in a tracker which does a forward pass through all frames and corrects body part identities with rules to preserve all reliable frames that display continuity in motion and realistic mouse skeletons. The next step is an EKF smoother based on a random walk state space model, with linearized priors implementing constraints on body-part distances as well as their vector angles. Smoothing is applied on frames that were unreliable based on rules from the previous step, and it greatly improves the accuracy of predictions for these frames. Measuring the prediction accuracy with Root Mean Squared Error (RMSE) on 40 manually labelled random frames, average errors for body parts are reduced to within 1.5 cm, while adequate tuning of parameters can further decrease the RMSE to less than 1 cm, sufficient for behavior analysis. These pose predictions are then imported to SimBA, so that social and freezing behavior bouts are classified with machine learning trained on annotations. Exported behavior bouts are then annotated according to whether the experiment mouse initiates the interaction, and they can be compared with fully manual annotations using a Gantt plot.

---

## Requirements

- R (>= 4.0 recommended)
- Python (3.9 for SLEAP, virtual environment 3.6 for SimBA)
- R packages:
  - `dplyr`, `tidyr`, `purrr`, `tibble`
  - `ggplot2`, `plotly` (visualization)
  - (optional) `gganimate` 
- Python packages: `sleap`, `pandas`, `h5py`, `numpy`

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
5. Run `Predictions_1.qmd` to complete Kalman smoothing to fill in body parts for missing frames and reduce MSFE of predictions.
6. Convert improved predictions to `.slp` format with `Prediction_import.py`, check for identity swaps and manually correct in a code chunk within `Predictions_1.qmd`.
7. Use `Wide.py` to transform the `.csv` file of improved predictions into SimBA ready format.
8. Import files to SimBA to extract features, train machine learning model for social interactions and obtain bout times.
9. Manually label approach / being approached for each bout to produce final gantt plots of behaviors.


## Main Logic for `Prior.qmd`:
Rule-based instance matching to ensure continuity in mouse identity over time.
If movement from last reliable frame / frame gap is above threshold, add motion penalty.
Reliability score: nll_dist (penalizes unrealistic body-part distances) + nll_ang (penalizes unrealistic angles) * w + motion_pen

One forward pass through data in time frame order, jointly match the body part candidates (8 combinations, fill NA with previous frame) with two tracked instances, accept candidates if reliability score is below threshold. 

Record reliable frames (where a body part moves and is accepted, about ¾ of mid-centers are tracked as reliable), decay skeleton scores for time since last reliable observation.
When a body part moves and is within the motion threshold, but isn’t accepted due to violating angle/distances, it is recorded but flagged as unreliable, allowing future frames to use it as candidates during tracking but will not be accepted as known data in the Kalman smoother.
When a body part has no valid coordinates to accept, the previous coordinates are retained for this frame.

Guardrails against identity swaps: Treat observations within a distance threshold as one, give penalty if the vector of two mid-centers makes a turn over 90 degrees or one mid-center gets too close to the other.

## Main Logic for `Predictions_1.qmd`:

Kalman filter: Use the reliable frames from tracking as known data, do smoothing and skeleton prior constraints for predictions. Core -- Random walk state-space model: 𝑦_𝑡=𝑠_𝑡+ 𝜀_𝑡 ,𝜀_𝑡 ~ 𝑖.𝑖.𝑑. 𝑁(0,𝑅)
𝑠_𝑡=𝑠_(𝑡−1)+ 𝜂_𝑡 ,𝜂_𝑡∼𝑖.𝑖.𝑑. 𝑁(0, 𝑄)

Add prior constraints: use Extended Kalman Filter to linearize the non-linear angle and distance constraints with Jacobians. Forward filtering step: allows the Kalman filter to work with streaming data Backward RTS smoothing: increases accuracy for offline predictions

𝑦_𝑡^𝑜𝑏𝑠=𝐻_𝑜𝑏𝑠 (𝑡) 𝑠_𝑡+𝜀_𝑡^𝑜𝑏𝑠,〖 𝜀〗_𝑡^𝑜𝑏𝑠~ 𝑁 (0, 𝑅_𝑜𝑏𝑠 (𝑡)) 𝑧_𝑡^𝑠𝑘𝑒𝑙≈〖𝐻_𝑠𝑘𝑒𝑙 (𝑡)𝑠〗_𝑡+ 𝜀_𝑡^𝑠𝑘𝑒𝑙,𝜀_𝑡^𝑠𝑘𝑒𝑙 ~𝑁(0, 𝑅_𝑠𝑘𝑒𝑙 (𝑡))

Combined: 〖 𝑧〗_𝑡= ((𝑦_𝑡^𝑜𝑏𝑠)¦(𝑧_𝑡^𝑠𝑘𝑒𝑙 ))=H_t s_t+ 𝜀_𝑡,𝜀_𝑡 ~ 𝑁(0, 𝑅_𝑡 ) H_t = ((𝐻_𝑜𝑏𝑠 (𝑡))¦(𝐻_𝑠𝑘𝑒𝑙 (𝑡))), 𝑅_𝑡=𝑏𝑙𝑜𝑐𝑘𝑑𝑖𝑎𝑔(𝑅_𝑜𝑏𝑠 (𝑡), 𝑅_𝑠𝑘𝑒𝑙 (𝑡))

The prior constraints enter the model through 𝐻_𝑠𝑘𝑒𝑙

After smoothing, obtain point estimates from smoothed posterior means and SD estimates from the smoothed posterior covariance matrix.

For frames that are reliable and have both neighbors moving within the motion threshold, assume that tracking is sufficiently accurate and directly use the tracked results as point predictions. 

For other frames, using a weighted average of the smoothed results and tracked results (0.8 * smoothed + 0.2 * tracked has good empirical results) 

Evaluate point predictions: computed mean squared forecast error (MSFE) from matching the predicted instances with the manual labels in the 40 labelled frames, tune parameters to reduce MSFE.


<img width="2553" height="180" alt="image" src="https://github.com/user-attachments/assets/54d5d525-5a4b-41b3-aea6-2693d2cd50ec" />


# Forward combos process
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

