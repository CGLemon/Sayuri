# SGF DownLoader

The tool to download SGF file from katago server.

## Requirements

* requests

## Usage

| Parameter      | Type              | Description                 |
| :------------: | :---------------: | :---------------:           |
| -g, --game     | int               | Number of downloading games.|
| -d, --directory| string            | The saving directory.       |
| -u, --url      | string            | Kata Go Server games URL.   |

Here is sample command.

    $ python3 main.py --game 1000 --dir saving

Here is sample URL.

    $ python3 main.py --game 1000 --dir saving -u https://katagotraining.org/networks/kata1/kata1-b60c320-s6220126976-d2925574164/training-games/

Others arguments please enter.

    $ python3 main.py -h
