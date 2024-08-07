#  Configuration File Explanation

## The Training Setting

The ```selfplay-setting.json``` controls the training process. Here are the parameters.

```
{
    "NeuralNetwork" : {
        "NNType" : "Residual",
        "MaxBoardSize" : 19,        # The max size in the self-play game. It is
                                    # OK if this value greater than training games.
                                    # But to set it as small size can improve the
                                    # training performance.

        "ResidualChannels" : 128,   # Channel size of residual.
        "PolicyExtract" : 24,       # Channel size of policy head.
        "ValueExtract" : 24,        # Channel size of value head.

        "Stack" : [
            "ResidualBlock",        # The 1st residual block. It is normal block.
            "ResidualBlock",        # The 2nd residual block. It is normal block.
            "ResidualBlock-SE",     # The 3rd residual block. It is block with SE module.
            "ResidualBlock",        # The 4th residual block.
            "ResidualBlock",        # The 5th residual block.
            "ResidualBlock-SE"      # The 6th residual block.
        ]
    },

    "Train" : {
        "UseGPU" : null,
        "Optimizer" : "SGD",
        "StepsPerEpoch" : 4000,      # Save the weight evey this steps.
        "ValidationSteps" : 100,
        "VerboseSteps" : 1000,
        "MaxStepsPerRunning" : 4000, # Will stop the training after this steps.
        "Workers" : 4,               # Number of data loader workers.
        "BatchSize" : 256,
        "BufferSize" : 524288,       # Bigger is better but it will use more memory. If your 
                                     # compute is only 32GB, you can set it as around 256000.

        "DownSampleRate" : 16,       # Bigger is better but may be slow down.
        "MacroFactor" : 1,
        "WeightDecay" : 1e-4,
        "NumberChunks" : 20000,      # Will load last X chunks. Default is 25 games for
                                     # each chunk.

        "LearningRateSchedule" : [
            [0,       1e-2]          # The format is [X, lr]. Will use the lr rate
                                     # after X stpes. You only need to change the lr
                                     # part in the reinforcement learning.
         ],

        "TrainDirectory" : "selfplay/data",
        "StorePath" : "workspace"
    }
}
```

## The Self-play Engine Configuration File

The ```selfplay-config.txt``` controls the self-play process. Here are the parameters.

```
--dirichlet-noise            # Enable Dirichlet noise in MCTS.

--random-moves-factor 0.08   # Do the random move if the move number is
                             # below the (X * intersections).

--random-opening-prob 0.75   # Play opening with high temperature policy in
                             # this probability.

--random-fastsearch-prob 0.75 # Play random move for reduce-playouts in this
                              # probability.

--komi-stddev 2.5            # Apply the random komi in the self-play
                             # games.

--cpuct-init 1.25
--lcb-reduction 0
--score-utility-factor 0.05

--selfplay-query bkp:9:7:0.9 # The format is boardsize-komi-prob.
                             # It means 90% games are 9x9 with komi 7.

--selfplay-query bhp:9:2:0.1 # The format is boardsize-handicap-prob.
                             # It means 10% 9x9 games is handicap. The
                             # max handicap size is 2.

--selfplay-query srs:area      # The self-play game will use area scoring and
--selfplay-query srs:territory # territory scoring.

--selfplay-query bkp:7:9:0.1 # It means 10% games are 7x7 with komi 9.

--playouts 150               # The main playouts.

--gumbel                     # Enable Gumbel search.

--gumbel-playouts-threshold 50 # Do the Gumbel search if the current playouts
                               # below this value. 

--always-completed-q-policy

--reduce-playouts 50         # 75% uses 50 playouts. Will disable any noise
                             # do not record the training data.
 
--reduce-playouts-prob 0.75  # 75% uses reduce-playouts.

--resign-playouts 75         # Use this playout if someone's winrate below
                             # threshold.

--resign-threshold 0.05      # If someone's winrate is below this value,
                             # will use the resign-playouts.

--resign-discard-prob 0.8    # Discard the training data in 80% when someome
                             # has already won the game.

--parallel-games 128         # Parallel games at the same time.
--batch-size 64              # Network evalutaion batch size.
--cache-memory-mib 400
--early-symm-cache
--first-pass-bonus

--root-policy-temp 1.1       # The policy softmax temperature of root node.  

--num-games 5000             # Self-play games per epoch.
```
