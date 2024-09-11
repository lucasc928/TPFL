# RAFL

This repository is the code of [Reputation-Driven Asynchronous Federated Learning for Enhanced Trajectory Prediction with Blockchain
](https://arxiv.org/abs/2407.19428). 

___
## License
This code is shared only for research purposes, and this cannot be used for any commercial purposes. 

___
## Training 
1. Prepare for the environment.
``` Bash
$ pip install pandas
$ pip install scipy
$ pip install gym
$ pip install stable-baselines3
```
2. Download the dataset. 
``` Bash
$ NGSIM: https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj/about_data
$ Apolloscape: https://apolloscape.auto/trajectory.html
```

3. Modify "data_root" in data_process.py and then run the script to preprocess the data. 
``` Bash
$ python data_process.py
```

4. Train the model.
``` Bash
$ python main.py
```
___
## Citation
Please cite our papers if you used our code. Thanks.
``` 
@misc{chen2024RAFL,
      title={Reputation-Driven Asynchronous Federated Learning for Enhanced Trajectory Prediction with Blockchain}, 
      author={Weiliang Chen and Li Jia and Yang Zhou and Qianqian Ren},
      year={2024},
      eprint={2407.19428},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.19428}, 
}
```
