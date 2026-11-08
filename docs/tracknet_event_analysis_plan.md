# TrackNet Event Analysis

TrackNet produces image-space shuttle observations, not semantic events such as “serve,” “winner,” or “landed in.” Its output can support event suggestions, but player tracking and human review remain important.

## Current support

| Stage | TrackNet contribution | Limit |
|---|---|---|
| 3A | Runs TrackNetV3 and exports one selected shuttle coordinate per frame, a CSV, and an optional preview. | A selected coordinate can switch to an adjacent court or miss an occluded shuttle. InpaintNet is not used by the pipeline. |
| 4 | Optionally uses a confident launch-origin cue to refine the server/receiver label after normal candidate generation and the centre-line pass. | It does not create, remove, or merge serve candidates; review in the Stage 4 UI. |
| 6 | Can refine the movement-based pause estimate when a sustained main-court shuttle cluster is nearby. | Weak or missing shuttle evidence keeps the player-movement estimate. |
| 7 | Provides shot-count candidates from the floor-line or optional clicked net-plane method. | Image-plane crossings are approximate and can be undercounted or include overlap; review and correct counts. |
| 9 | Adds a tentative stationary shuttle location for review. | Pickup-based landing-side correction remains authoritative. |
| 10 | Includes the available shuttle, shot-count, and rally-end evidence in the report. | These measurements are clues, not proof of a winner or cause. |
| 11 | Can draw a filtered shuttle trail and live shot counter. | Adjacent-court detections and gaps can still appear. |

## What is not established

The current pipeline does not use TrackNet alone to determine a point winner, confirm an in/out call, find an exact landing zone, estimate true 3D flight, or identify shot type. The floor homography applies to points on the floor; it cannot directly map an airborne image point to a landing coordinate. A candidate stationary point is not a confirmed landing.

## Pilot record

The short pilot found that raw TrackNet detections followed the rally more plausibly than InpaintNet-repaired trails. The pilot also showed shuttle detections on adjacent courts. The integrated pipeline therefore uses raw detections, applies court-region filtering as a soft cue, and keeps downstream events reviewable.

For setup and the pilot procedure, see [TrackNet Pilot Notes](tracknet_pilot_plan.md). For commands, see [Command Pipeline](command_pipeline.md). Any future event classifier should be validated on short clips with reviewed serve, rally-end, landing-side, and shot-count labels before it influences the score.
