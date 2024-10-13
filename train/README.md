# Train Code Usage

## Supervised Learning

You should prepare the training data and config json file. Rewrite the ```TrainDirectory``` as your training data path. You may ask how to find out the data? The only way is you need to generate the data by yourself via RL script. Please go to the bash to see the detail. After you run RL loop, the generated data should be at the ```selfplay/tdata```. Then type the following script

    $ python3 torch/main.py -j sl-setting.json


## Python Engine

PySayuri is independent GTP engine without any CPP source code. PySayuri provides a wrapper to excute network for developer. All functions are written in ```Agent``` class. You could be more easy to import code into your project. The checkpoint models are released after 4th main run in the [page](../docs/MODEL.md), so you don't need to run RL loop again. For quick start, please type

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

It is experimental code and based on PySayuri. The visualization tool can print feature map of policy/value head. Please download the checkpoint model first. Then type the following script. The result pictures will be saved in the ```result```

    $ python3 torch/visualize.py -c model.pt -r result


If you want to select a specific SGF file, please add the argument ```--sgf```.

    $ python3 torch/visualize.py -c model.pt -r result --sgf sgf-file
