# Reinforcement Learning

## Requirements

* Be sure that you had built the engine. The engine should be in the ```build``` directory.
* PyTorch 1.x (for python)
* Numpy (for python)

## Simple Usage

There three bash files. The ```setup.sh``` will do the initialization. The ```selfplay.sh``` will do the self-play loop. The ```replace.sh``` will train a new network.

    $ cp -r bush selfplay-course
    $ cd selfplay-course
    $ bash setup.sh -s ..
    $ bash selfplay.sh


There two important parameters ```GAMES_PER_EPOCH``` and ```MAX_TRAINING_EPOCHES``` in selfplay.sh. They control the totally played games.

## The Training Setting

The ```selfplay-setting.json``` controls the training process. Here are the parameters.

```
{
    "NeuralNetwork" : {
        "NNType": "Residual",
        "MaxBoardSize": 19,         # The max size in the self-play game. It is
                                    # OK if this value greater than training games.
                                    # But set a small size can improve the training
                                    # performance

        "InputChannels": 38,
        "ResidualChannels": 128,    # channel size
        "PolicyExtract": 24,        # channel size of policy head
        "ValueExtract": 24,         # channel size of value head
        "ValueMisc": 5,

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
        "UseGPU": null,
        "Optimizer": "SGD",
        "StepsPerEpoch" : 4000,      # Save the weight evey this steps.
        "ValidationSteps": 100,
        "VerboseSteps" : 1000,
        "MaxStepsPerRunning": 4000,  # Will stop the training after this steps.
        "Workers": 4,                # Number of data loader worker.
        "BatchSize": 256,
        "BufferSize" : 524288,
        "DownSampleRate": 16,
        "MacroFactor": 1,
        "WeightDecay": 1e-4,
        "NumberChunks" : 20000,      # Will load last X chunks.

        "LearningRateSchedule": [
            [0,       1e-2]          # The format is [X, lr]. Will use the lr rate
                                     # after X stpes.
         ],

        "FixUpBatchNorm": false,
        "TrainDirectory": "selfplay/data",
        "StorePath" : "workspace"
    }
}
```

## The Self-play Engine Config

The ```selfplay-config.txt``` controls the self-play process. Here are the parameters.

```
--noise                      # Enable Dirichlet noise in MCTS.

--random-moves-factor 0.08   # Do the random move if the move number is
                             # below the (X * intersections).

--komi-variance 2.5          # Apply the random komi in the self-play
                             # games.

--cpuct-init 1.25
--lcb-reduction 0
--score-utility-factor 0.05
--lcb-utility-factor 0.05
--completed-q-utility-factor 0.05

--selfplay-query bkp:9:7:0.9 # The format is boardsize-komi-prob.
                             # It means 90% games are 9x9 with komi 7.

--selfplay-query bhp:9:2:0.1 # The format is boardsize-handicap-prob.
                             # It means 10% 9x9 games is handicap. The
                             # max handicap size is 2.

--selfplay-query bkp:7:9:0.1 # It means 10% games are 7x7 with komi 9.

--playouts 150               # The default playouts.

--gumbel                     # Enable Gumbel search.

--gumbel-playouts 50         # Do the Gumbel search if the current playous
                             # below this. 

--always-completed-q-policy

--reduce-playouts 100        # 85% uses 100 playouts 
--reduce-playouts-prob 0.85

--resign-playouts 85
--resign-threshold 0.02      # If someone's winrate is below this value,
                             # will use the resign-playouts.

--parallel-games 128         # Parallel games at the same time.
--batch-size 64
--cache-memory-mib 400
--early-symm-cache
--first-pass-bonus
```
