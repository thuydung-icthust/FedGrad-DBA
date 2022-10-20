# Introduction
In this repository, code is for our implementation of FedGrad under trigger-based attacks.\
We evaluate FedGrad under two datasets: MNIST and CIFAR-10 under: unconstrained backdoor attack, constrain-and-scale backdoor attack and DBA.\
The implementation of these experiments is conducted by customizing the original implementation provided by an ICLR 2020 paper [DBA: Distributed Backdoor Attacks against Federated Learning](https://openreview.net/forum?id=rkgyS0VFvr).
## Installation Guide
This repository is employed to evaluate the effectiveness of FedGrad under trigger-based backdoor attacks on FL.
### Installation
- Pytorch (on the official link: https://pytorch.org/)
- Required packages on ```requirements.txt``` file.
- prepare the pretrained model:
    Pretrained clean models for attack can be downloaded from [Google Drive](https://drive.google.com/file/d/1wcJ_DkviuOLkmr-FgIVSFwnZwyGU8SjH/view?usp=sharing). 
    Then copy downloaded "saved_models.zip" to FedGrad-DBA repo && unzip.
- MNIST and CIFAR will be automatically download if experiments are run
### Experiment running
- run experiments for all the yaml file on folder ```experiments_yaml/AISTATS2023``` with the following command:
    ```python dba.py --params [yaml_file]```
    
    (e.g., ```python dba.py --params experiments_yaml/AISTATS2023/fedavg_1.1_params.yaml```)
## Acknowledgement 
We would like to acknowledge the following open-source projects:
- ICLR 2020 paper [DBA: Distributed Backdoor Attacks against Federated Learning](https://openreview.net/forum?id=rkgyS0VFvr)
- [ebagdasa/backdoor_federated_learning](https://github.com/ebagdasa/backdoor_federated_learning)
- [krishnap25/RFA](https://github.com/krishnap25/RFA)
- [DistributedML/FoolsGold](https://github.com/DistributedML/FoolsGold)
