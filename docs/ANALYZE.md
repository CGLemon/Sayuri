## Analysis Commands

The analysis Commands are useful for the modern GTP interface tool, like Sabaki. It shows the current win-rate, best move and the other informations. The engine supports the following GTP analysis commands.

  * `analyze, genmove_analyze [player (optional)] [interval (optional)] ...`
      * The behavior is same as ```lz-analyze```, ```lz-genmove_analyze```.

  * `lz-analyze, lz-genmove_analyze [player (optional)] [interval (optional)] ...`
      * Extension GTP commands of ```lz-analyze``` and ```lz-genmove_analyze```. Support the ```info```, ```move```, ```visits```, ```winrate```, ```prior```, ```lcb```, ```order```, ```pv```, ```scoreLead``` labels. More detail to see [KataGo GTP Extensions](https://github.com/lightvector/KataGo/blob/master/docs/GTP_Extensions.md).


  * `kata-analyze, kata-genmove_analyze [player (optional)] [interval (optional)] ...`
      * Subset of ```kata-analyze``` and ```kata-genmove_analyze```. Support the ```info```, ```move```, ```visits```, ```winrate```, ```prior```, ```lcb```, ```order```, ```pv```, ```scoreLead``` labels. More detail to see [KataGo GTP Extensions](https://github.com/lightvector/KataGo/blob/master/docs/GTP_Extensions.md).


  * Optional Keys
      * All analysis commands support the following keys.
      * ```interval <int>```: Output a line every this many centiseconds.
      * ```minmoves <int>```: There is no effect.
      * ```maxmoves <int>```: Output stats for at most N different legal moves (NOTE: Leela Zero does NOT currently support this field);
      * ```avoid PLAYER VERTEX,VERTEX,... UNTILDEPTH```: Prohibit the search from exploring the specified moves for the specified player, until ```UNTILDEPTH``` ply deep in the search.
      * ```allow PLAYER VERTEX,VERTEX,... UNTILDEPTH```: Equivalent to ```avoid``` on all vertices EXCEPT for the specified vertices. Can only be specified once, and cannot be specified at the same time as ```avoid```.
      * ```ownership True```: Output the predicted final ownership of every point on the board.
      * ```movesOwnership True```: Output the predicted final ownership of every point on the board for every individual move.
