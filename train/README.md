# Train Code Usage

## Supervised Learning


To begin training, first prepare your training data and a configuration JSON file. You will need to update the ```TrainDirectory``` entry in your configuration file to point to your training data path. You may ask how to find out the data?  The training data must be generated through the Reinforcement Learning (RL) script. For detailed instructions on data generation, please refer to the relevant section in the [bash](../bash/README.md) directory. After running the RL loop, the generated data will be located in selfplay/tdata. 

Once your data is ready, execute the training script

    $ python3 torch/train.py -j sl-setting.json


## Python Engine

PySayuri is an independent GTP (Go Text Protocol) engine, implemented entirely in Python without any C++ source code. It offers a convenient wrapper for developers to execute its neural network. All core functionalities are encapsulated within the ```Agent``` class, making it easy to integrate PySayuri's code directly into your projects. Checkpoint models are released starting from the 4th Run and can be found on this [page](../docs/MODEL.md). This means you do not need to perform the Reinforcement Learning (RL) loop yourself.

To get started quickly, simply type

    $ python3 torch/pysayuri.py -c model.pt -p 100 --use-swa

More arguments descript are here.

- Some useful arguments
    - ```-c, --checkpoint```: The path of checkpoint.
    - ```-p, --playouts```: The number of playouts. Larger is stronger.
    - ```-v, --verbose```: Dump the MCTS search verbose.
    - ```--use-swa```: When loading the checkpoint, will use SWA model inseat of basic mode.
    - ```--use-gpu```: Will use the GPU if GPU is valid.
    - ```--scoring-rule```: Should be one of area/territory. The area is Chinese-like Rules. The territory is Japanese-like Rules.

## Visualization Tool

This experimental tool, built upon PySayuri, allows you to visualize the feature maps of the policy and value heads. First, please download the desired checkpoint model. Then, execute the following script. The result pictures will be saved in the ```result```


    $ python3 torch/visualize.py -c model.pt -r result

To process a specific SGF file, include the ```----sgf```-- argument. Please type

    $ python3 torch/visualize.py -c model.pt -r result --sgf sgf-file
