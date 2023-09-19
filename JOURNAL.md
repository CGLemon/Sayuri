## Journal

### About the ancient technology

(August, 2022)

Before the AlphaGo (2016s), the most of state-of-the-art computer Go combine the MCTS and MM (Minorization-Maximization). Crazy Stone and Zen use that. Or combining the MCTS and SB (Simulation Balancing). The Eric (predecessor of AlphaGo) and Leela use that. Ray, one of the strongest open source Go engine before AlphaGo, writed by Yuki Kobayashi which is based on the MM algorithm. I am surprised that it can play the game well without much human knowledge and Neural Network. What's more, it can beat high level Go player on 9x9 if we provide it enough computation. But thanks for deep learning technique, the computer Go engine is significantly stronger than before. Sayuri can beat the Ray (v10.0) on 19x19 with only policy network. This result shows the advantage of Neural Network technology.

Although the Neural Network based engines are more powerful, I still recommend that you try some engine with non Neural Network and feel the power of ancient technology. Here is the list.

* [Leela](https://www.sjeng.org/leela.html), need to add the option ```--nonets``` to disable DCNN.
* [Pachi](https://github.com/pasky/pachi), need to add the option ```--nodcnn``` to disable DCNN.
* [Ray](https://github.com/kobanium/Ray), may be strongest open source engine before the 2016s.

I had implemented this ancient technique. Merge the MM patterns based and the DCNN based technique to provide widely dynamic strength.

* This part had been removed since August, 2023. The engine does not support no-DCNN mode now. Only remain some relevant functions.

### The Gumbel learning

(November, 2022)

On the 2022 CGF Open, the Ray author, Yuki Kobayashi, implemented a new algorithm called Gumbel learning. it is a effective trick for AlphaZero and it guarantees to improve policy with low playouts. As far as I know, Ray is the first successful superhuman level engine with Gumbel learning on 19x19. Inspired by Ray, I decide to implement this ideal in my project.

* [Policy improvement by planning with Gumbel](https://www.deepmind.com/publications/policy-improvement-by-planning-with-gumbel)
* [Ray's apeal letter for UEC 14](https://drive.google.com/file/d/1yLjGboOLMOryhHT-aWG_0zAF-G7LDcTH/view)

After playing two million games (May, 2023), the strengh reached pro level player. I believe that Sayuri's performance successfully approaches the early KataGo's (g65).

### Improve the network performance

(February, 2023)

The Ray author, Yuki Kobayashi, proposed three points which may improve my network performance. Here are list.

* The half floating-point.
* The NHWC format.
* Bottleneck network, It may improve 30% speed without losing accuracy.

KataGo also proposed a variant bottleneck and said it could significantly improve the performance. This result shows the advance of these kinds of structure. However in my recent testing (March, 2023), bottleneck is not effective on the 10x128 and 15x192 network. And seem that there are more blind spots in bottleneck because the 1x1 kernel may compress the board information. I will check it again.


### Break the SL pipe and next step

(July, 2023)

Thanks for [shengkelong](https://github.com/shengkelong)'s experiment. The experiment gives the project some better parameters and ideas. Some ideas are about RL improvement which is generating the training data from the self-play. I think it is time to break the SL pipe because the current RL weights is stronger than any SL weights which I train. Breaking the SL pipe can be more easy to rewrite the RL pipe.

### New Run

(August, 2023)

Base on the experience of first run. I select some better hyper-parameters. Look like that the early phase of training is better than KataGo g65 and g104 (g170 is unknown) and we don't need to tune many hyper-parameters. I think this project could be better than current KataGo in early phase after tuning more hyper-parameters.

