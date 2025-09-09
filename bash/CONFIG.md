#  Configuration File Explanation

## The Training Setting

The selfplay-setting.json file defines the key parameters for the neural network structure and training process. It includes settings for the neural network architecture, such as the number of residual blocks, channel sizes, and activation functions, as well as training configurations like batch size, learning rate schedule, and data buffer size. These settings allow flexible adjustment of training behavior and resource usage, making it possible to balance training efficiency and model performance according to the available hardware.

```
{
    "NeuralNetwork": {
        "NNType": "Residual",
        "MaxBoardSize": 19,          # The maximum board size for self-play games.
                                     # It is acceptable to set this value larger than
                                     # the training board size, but it may hurt performance.

        "ResidualChannels": 128,     # The number of channels in the residual blocks.
        "PolicyHeadChannels": 24,    # The number of channels in the policy head.
        "ValueHeadChannels": 24,     # The number of channels in the value head.

        "SeRatio": 4,                # Squeeze ratio of the SE (Squeeze-and-Excitation) module.
        "PolicyHeadType": "Normal",  # Should be one of [Normal, RepLK].
        "Activation": "mish",        # Should be one of [relu, swish, mish].

        "Stack": [                    # Each block should be one of [
                                      # ResidualBlock, BottleneckBlock,
                                      # NestedBottleneckBlock, MixerBlock ]

            "ResidualBlock",          # The first residual block (standard).
            "ResidualBlock",          # The second residual block (standard).
            "ResidualBlock-SE",       # The third residual block with an SE module.
            "ResidualBlock",          # The fourth residual block (standard).
            "ResidualBlock",          # The fifth residual block (standard).
            "ResidualBlock-SE"        # The sixth residual block with an SE module.
        ]
    },

    "Train": {
        "UseGPU": null,              # Set to null for automatic GPU selection.
        "Optimizer": "SGD",
        "StepsPerEpoch": 4000,       # Save weights every this many steps.
        "ValidationSteps": 100,
        "VerboseSteps": 1000,
        "MaxStepsPerRunning": 4000,  # Stop training after this many steps.
        "Workers": 4,                # Number of data loader workers.
        "BatchSize": 256,
        "BufferSize": 524288,        # Larger values are better but consume more memory.
                                     # For systems with 32GB RAM, around 256000 is recommended.

        "DownSampleRate": 16,        # Larger values improve data freshness but may slow down training.
        "MacroFactor": 1,
        "WeightDecay": 1e-4,

        "ChunksIncreasingC": 5000,   # Gradually increase the replay buffer size after at least
                                     # this many chunks. If set to null, will load "NumberChunks".

        "NumberChunks": 20000,       # Maximum number of recent chunks to load.
                                     # By default, each chunk contains one game.

        "PolicySurpriseFactor": 0.5, # A factor affecting sample selection rates.

        "LearningRateSchedule": [
            [0, 1e-2]                # Format: [step, learning rate]. Learning rate changes after
                                     # reaching the specified step. Only modify the learning rate
                                     # for reinforcement learning.
        ],

        "TrainDirectory": "selfplay/data",
        "StorePath": "workspace"
    }
}
```

## The Self-play Engine Configuration File

The selfplay-config.txt file controls the behavior of the self-play engine, which is responsible for generating training data through automated games. It configures options for the Monte Carlo Tree Search (MCTS), game rules (such as komi and board size), randomness in play styles, playout counts, and resignation behavior. This setup ensures diverse and efficient data generation, enabling the model to learn from a wide range of game scenarios while optimizing computational costs through techniques like reduced playouts and parallel self-play.

```
--dirichlet-noise               # Enable Dirichlet noise in MCTS.

--random-moves-factor 0.08      # Play a random move if the move number is
                                # less than (factor × number of intersections).

--random-opening-prob 0.75      # Probability of using a high-temperature policy
                                # during the opening phase.

--random-fastsearch-prob 0.75   # Probability of playing a random move during
                                # fastsearch-playout searches.

--komi-stddev 2.5               # Standard deviation for random komi in self-play games.

--cpuct-init 0.5
--lcb-reduction 0
--score-utility-factor 0.05

--selfplay-query bkp:9:7:0.9    # Format: boardsize-komi-probability. 90% of games are 9×9 with komi 7.

--selfplay-query bkp:7:9:0.1    # Format: boardsize-komi-probability. 10% of games are 7×7 with komi 9.

--selfplay-query bhp:9:2:0.1    # Format: boardsize-handicap-probability. 10% of 9×9 games use handicap, up to 2 stones.

--selfplay-query srs:area       # Self-play games use area scoring (Chinese rules).

--selfplay-query srs:territory  # Self-play games use territory scoring (Japanese rules).

--playouts 150                  # Main number of playouts per move.

--kldgain-per-node 0.000004     # Minimum KLD gain required to continue search. If the gain
                                # falls below this value, the search is stopped early.

--kldgain-interval 50           # Frequency (in visits) to check KLD gain during search. A
                                # lower value means more frequent checks, higher value means
                                # fewer checks.

--gumbel                        # Enable Sequential Halving with Gumbel search.

--gumbel-playouts-threshold 50  # Use Gumbel search when current playouts
                                # are below this threshold.

--always-completed-q-policy

--fastsearch-playouts 50        # In 75% of games, use 50 playouts without noise
                                # and do not save training data. Should be lower
                                # than main playouts.

--fastsearch-playouts-prob 0.75 # Probability of using reduced playouts.

--resign-playouts 32            # Use this number of playouts if a player's win rate
                                # falls below the resign threshold. Should be lower
                                # than fastsearch-playouts.

--resign-threshold 0.05         # Resign if a player's win rate falls below this threshold.

--resign-discard-prob 0.8       # Discard 80% of training data when the game
                                # has a clear winner.

--parallel-games 64             # Number of self-play games to run in parallel.

--cache-memory-mib 400          # Cache memory size in MiB.

--early-symm-cache
--first-pass-bonus

--root-policy-temp 1.1          # Temperature for the root node policy softmax.

--num-games 5000                # Number of self-play games per epoch.
```
