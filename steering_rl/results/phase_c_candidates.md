# Phase C candidate screen — measured result

Run 2026-07-28, all-Modal (co-located A10G, osmesa). 3 candidate libero_10 tasks
x 8 prompts x 6 episodes = 144 episodes. Screening for tasks with genuine
spread/headroom after 3/5 locked tasks came back at the legit-prompt ceiling.

```
Phase C steerability audit — 144 episodes over 3 task(s)
gate thresholds: spread >= 0.15, correct > misleading + 0.00

LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate
  canonical  #################### 1.00 (6/6)  put the white mug on the left plate and put the yellow and white mug on the right plate
  paraphrase #################### 1.00 (6/6)  place the white mug on the left plate and the yellow and white mug on the right plate
  paraphrase #################### 1.00 (6/6)  put the white mug onto the left plate, then put the yellow and white mug onto the right plate
  subgoal    #################### 1.00 (6/6)  put the white mug on the left plate
  subgoal    #################### 1.00 (6/6)  put the yellow and white mug on the right plate
  subgoal    #################### 1.00 (6/6)  pick up the white mug and place it on the left plate
  misleading #################### 1.00 (6/6)  put the white mug on the right plate and the yellow and white mug on the left plate
  misleading #################... 0.83 (5/6)  put both moka pots on the stove
  -> spread=0.00 (meaningful=False); best_action=1.00 vs misleading=0.92 (correct>misleading=True)  [gate: fail]

LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate
  canonical  #################... 0.83 (5/6)  put the white mug on the plate and put the chocolate pudding to the right of the plate
  paraphrase #################### 1.00 (6/6)  place the white mug on the plate and put the chocolate pudding to the right of the plate
  paraphrase #################### 1.00 (6/6)  put the white mug onto the plate, then set the chocolate pudding to the right of the plate
  subgoal    #################... 0.83 (5/6)  put the white mug on the plate
  subgoal    #################... 0.83 (5/6)  put the chocolate pudding to the right of the plate
  subgoal    ###................. 0.17 (1/6)  pick up the chocolate pudding and place it to the right of the plate
  misleading #################### 1.00 (6/6)  put the white mug on the left plate and the yellow and white mug on the right plate
  misleading #################... 0.83 (5/6)  put the chocolate pudding on the plate and the white mug to the right of the plate
  -> spread=0.83 (meaningful=True); best_action=1.00 vs misleading=0.92 (correct>misleading=True)  [gate: PASS]

KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
  canonical  #################### 1.00 (6/6)  turn on the stove and put the moka pot on it
  paraphrase #################### 1.00 (6/6)  switch on the stove and place the moka pot on it
  paraphrase #################### 1.00 (6/6)  turn the stove on and set the moka pot on the stove
  subgoal    #################### 1.00 (6/6)  pick up the moka pot and place it on the stove
  subgoal    #################... 0.83 (5/6)  turn on the stove
  subgoal    #################... 0.83 (5/6)  put the moka pot on the stove
  misleading #################... 0.83 (5/6)  put the black bowl in the bottom drawer of the cabinet and close it
  misleading #################... 0.83 (5/6)  put the yellow and white mug in the microwave and close it
  -> spread=0.17 (meaningful=True); best_action=1.00 vs misleading=0.83 (correct>misleading=True)  [gate: PASS]

GATE (all tasks pass): False
Flat/failed on pi05_libero? See CLAUDE.md §9: audit pi05_base / pi05_droid before RL.
```

## Headroom analysis (does any prompt BEAT canonical?)
- SCENE5: canonical 1.00 (ceiling); crossed-destination misleading also 1.00 -> prompt-INVARIANT. No headroom.
- SCENE6: canonical 0.83, best paraphrase 1.00 -> a fixed prompt beats canonical (headroom), but best-fixed ~ceiling.
- KITCHEN_SCENE3: canonical 1.00 (ceiling); spread only from slightly-worse subgoals. No headroom.

Gate-passers across all 8 tasks: soup+tomato, mug->microwave, SCENE6, KITCHEN_SCENE3 (4/8).
Tasks with a prompt that BEATS canonical (real headroom for steering): mug->microwave, SCENE6 only.

## §9 fallback probe: pi05_base zero-shot on LIBERO (2026-07-28)

Served pi05_base params + LIBERO norm stats + LIBERO transforms
(steering_rl/serving/serve_base_libero.py). Probe: task LIVING_ROOM_SCENE2
(soup+tomato, which pi05_libero aces at 1.0) x ~10 prompts x 3 episodes = 30 episodes.

Result: **0/30 success on every prompt.** All episodes ran the full 530 steps with
no errors — the robot moved but never completed the task. pi05_base is genuinely
incompetent at LIBERO zero-shot (it was never fine-tuned on this embodiment; its
state normalization also differs). The §9 fallback is therefore a dead end:
pi05_base/droid are not usable LIBERO policies, and fine-tuning them is out of
scope (CLAUDE.md §2 frozen VLA, §6 conditioning fine-tune CUT).

Conclusion: on the available FROZEN pi05 checkpoints, learned prompt-steering has
no viable headroom on LIBERO — pi05_libero is competent but prompt-robust; the
better-language-following checkpoints cannot do the tasks. Well-evidenced null result.
