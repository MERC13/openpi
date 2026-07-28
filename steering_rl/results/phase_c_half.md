# Phase C steerability audit — measured result (half protocol)

Run 2026-07-27 on Modal (deployed `steering-rl-audit`, A10G, osmesa).
500 episodes = 5 locked tasks × ~10 prompts × 10 episodes (half of §7's 20).
Numbers are measured (CLAUDE.md §2), rendered by `steering_rl/libero/report.py`.

```
Phase C steerability audit — 500 episodes over 5 task(s)
gate thresholds: spread >= 0.15, correct > misleading + 0.00

LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket
  canonical  #################### 1.00 (10/10)  put both the alphabet soup and the tomato sauce in the basket
  paraphrase #################### 1.00 (10/10)  place both the alphabet soup and the tomato sauce into the basket
  paraphrase #################### 1.00 (10/10)  move the alphabet soup and the tomato sauce into the basket
  paraphrase #################### 1.00 (10/10)  pick up the alphabet soup and the tomato sauce and put them in the basket
  paraphrase ##################.. 0.90 (9/10)  put the alphabet soup and the tomato sauce in the basket
  subgoal    #################### 1.00 (10/10)  put the alphabet soup in the basket
  subgoal    #################### 1.00 (10/10)  put the tomato sauce in the basket
  subgoal    ################.... 0.80 (8/10)  pick up the tomato sauce and place it in the basket
  misleading .................... 0.00 (0/10)  put the black bowl in the bottom drawer of the cabinet and close it
  misleading .................... 0.00 (0/10)  put both the cream cheese box and the butter in the basket
  -> spread=0.20 (meaningful=True); best_action=1.00 vs misleading=0.00 (correct>misleading=True)  [gate: PASS]

LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket
  canonical  #################### 1.00 (10/10)  put both the alphabet soup and the cream cheese box in the basket
  paraphrase #################### 1.00 (10/10)  place both the alphabet soup and the cream cheese box into the basket
  paraphrase #################### 1.00 (10/10)  put the alphabet soup and the cream cheese box in the basket
  paraphrase #################### 1.00 (10/10)  move the alphabet soup and the cream cheese box into the basket
  paraphrase #################### 1.00 (10/10)  pick up the alphabet soup and the cream cheese box and put them in the basket
  subgoal    #################### 1.00 (10/10)  put the alphabet soup in the basket
  subgoal    #################### 1.00 (10/10)  put the cream cheese box in the basket
  subgoal    #################### 1.00 (10/10)  pick up the cream cheese box and place it in the basket
  misleading ##################.. 0.90 (9/10)  put both the cream cheese box and the butter in the basket
  misleading ##############...... 0.70 (7/10)  put the yellow and white mug in the microwave and close it
  -> spread=0.00 (meaningful=False); best_action=1.00 vs misleading=0.80 (correct>misleading=True)  [gate: fail]

LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket
  canonical  #################### 1.00 (10/10)  put both the cream cheese box and the butter in the basket
  paraphrase #################### 1.00 (10/10)  place both the cream cheese box and the butter into the basket
  paraphrase #################### 1.00 (10/10)  put the cream cheese box and the butter in the basket
  paraphrase #################### 1.00 (10/10)  move the cream cheese box and the butter into the basket
  paraphrase #################### 1.00 (10/10)  pick up the cream cheese box and the butter and put them in the basket
  subgoal    #################### 1.00 (10/10)  put the cream cheese box in the basket
  subgoal    #################### 1.00 (10/10)  put the butter in the basket
  subgoal    #################### 1.00 (10/10)  pick up the butter and place it in the basket
  misleading ##########.......... 0.50 (5/10)  put the black bowl in the bottom drawer of the cabinet and close it
  misleading .................... 0.00 (0/10)  put both the alphabet soup and the tomato sauce in the basket
  -> spread=0.00 (meaningful=False); best_action=1.00 vs misleading=0.25 (correct>misleading=True)  [gate: fail]

KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
  canonical  ##################.. 0.90 (9/10)  put the black bowl in the bottom drawer of the cabinet and close it
  paraphrase #################### 1.00 (10/10)  place the black bowl into the bottom drawer of the cabinet and close the drawer
  paraphrase #################### 1.00 (10/10)  put the black bowl into the bottom drawer and close it
  paraphrase #################### 1.00 (10/10)  stow the black bowl in the bottom drawer of the cabinet and close the drawer
  paraphrase ##################.. 0.90 (9/10)  move the black bowl into the cabinet's bottom drawer and shut it
  subgoal    #################### 1.00 (10/10)  put the black bowl in the bottom drawer of the cabinet
  subgoal    #################### 1.00 (10/10)  close the bottom drawer of the cabinet
  subgoal    #################### 1.00 (10/10)  pick up the black bowl and place it in the bottom drawer
  misleading #################### 1.00 (10/10)  put the yellow and white mug in the microwave and close it
  misleading ##################.. 0.90 (9/10)  put both the alphabet soup and the tomato sauce in the basket
  -> spread=0.10 (meaningful=False); best_action=1.00 vs misleading=0.95 (correct>misleading=True)  [gate: fail]

KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it
  canonical  ##################.. 0.90 (9/10)  put the yellow and white mug in the microwave and close it
  paraphrase ##################.. 0.90 (9/10)  place the yellow and white mug into the microwave and close the microwave
  paraphrase ##################.. 0.90 (9/10)  put the yellow and white mug into the microwave and shut the door
  paraphrase ##################.. 0.90 (9/10)  move the yellow and white mug into the microwave and close it
  paraphrase ##################.. 0.90 (9/10)  stow the yellow and white mug in the microwave and close the door
  subgoal    #################### 1.00 (10/10)  put the yellow and white mug in the microwave
  subgoal    ##################.. 0.90 (9/10)  pick up the yellow and white mug and place it in the microwave
  subgoal    ##############...... 0.70 (7/10)  close the microwave
  misleading ############........ 0.60 (6/10)  put the black bowl in the bottom drawer of the cabinet and close it
  misleading ####................ 0.20 (2/10)  put both the cream cheese box and the butter in the basket
  -> spread=0.30 (meaningful=True); best_action=1.00 vs misleading=0.40 (correct>misleading=True)  [gate: PASS]

GATE (all tasks pass): False
Flat/failed on pi05_libero? See CLAUDE.md §9: audit pi05_base / pi05_droid before RL.
```
