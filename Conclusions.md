# Conclusions

## Step 1: Recreating AlphaZero

- Successfully recreated AlphaZero model and training loop
- Created a highly-optimized infinite training loop, consisting of self-play, training, evaluation, and model promotion.
- Integrated Weights and Biases for hyperparameter tuning sweeps, and general logging and monitoring.
- Implemented a GUI for playing against the model.
- Implemented a 3rd party model evalution based on a pre-configured stockfish model.

## Step 2: Improving the Model

- Attempted to improve the model by implementing some of the advancements in the field that have been discovered since the original AlphaZero paper.
    - Converted from a single policy head to multiple policy heads with a gating mechanism, in order to mimic a rough MoE architecture.
    - Additionally implemented a separate head for legality, in order to separate the the duties of learning the best move from a given board, and learning which moves are legal.
    - Added an attention mechanism to the value head, in order to allow the model to focus on the most important features of the board.

### Results

- Preliminary testing was not as successful as I'd hoped, but more work is necessary to properly evaluate either model. I changed a few other aspects of the pipeline in the shift to V2, besides the models, so more direct testing between models is necessary as well.
- I honestly need much more compute to properly train any of these. I'm working with a mid-level gaming PC.

#### Details

- It takes me a little over 24 hours to run each generation of the training loop.
- I ran the V1 model for about 25 generations, and the V2 model for about 15 generations. I stopped because I realized a deeper issue, which I will address later in this section. Also, my room was getting too hot, since my computer was running at full capacity the entire time and it was July.
- The V1 gen 25 model tested better than the V2 gen 15 model (somewhat predictably) but neither tested particularly well. Neither was ever able to beat me, and I'm not good at chess. I don't actually play at all, I just thought this was a fun project. 
    - They were both able to play intelligibly, but not well. To their credit, both were able to force a draw against me at least once in roughly 20 games. It's a very rough test.
- I did not test the V1 gen 15 model. I'm realizing as I'm writing this that I do have it saved in W&B, but I don't really care to dig it up and test it because...
- I realized that when I upgraded the model I also increased its size. Because of this it was running slower, so when I ran the MCTS algorithm for the V2 model it was able to query the model less often. This is probably a much more significant problem than I initially realized.

## Next Steps:

- Again, I simply need more compute to really proceed with this project.
- My idea for next steps is inspired by the recent AlphaEvolve paper. I want to essentially make a model zoo, where I can train multiple different architures and compare them against each other.
- I'd like to test out each of my innovations from step 2 separately, to see which ones are most effective.
- We want the model to be as small as possible, so that we can query it as often as possible. However, I'll likely have to increase the size somewhat in order to accommodate the additional mechanisms that I want to add. This "model zoo" idea seems like a good way to determine the relative effectiveness of each innovation.

---

- Another idea I had was to skip the self-play portion of the training loop, and instead use a pre-configured engine to generate games. This would speed up the process significantly. The original AlphaZero team already proved that it's possible to learn chess well purely from self-play. It's not an essential part of the process and would essentially allow me to skip the first 100+ generations of the training loop.
- As I'm writing this I'm kind of talking myself into continuing this project, but I need to be done with it and move to something else. It's still true that I need more compute to train and evaluate a bunch of models in a reasonable amount of time, even if I cheat and generate games with a pre-configured engine.