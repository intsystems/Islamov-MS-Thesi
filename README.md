# Distributed Newton-Type Methods with Communication Compression and Bernoulli Aggregation

## Author: Islamov Rustem
## Supervisor: Strijov Vadim

# Abstract

Despite their high computation and communication costs, Newton-type methods remain an appealing option for distributed training due to their robustness against ill-conditioned convex problems. In this work, we study communication compression and aggregation mechanisms for curvature information in order to reduce these costs while preserving theoretically superior local convergence guarantees. We prove that the recently developed class of three point compressors (3PC) of Richtarik et al. [2022] for gradient communication can be generalized to Hessian communication as well. This result opens up a wide variety of communication strategies, such as contractive compression and lazy aggregation, available to our disposal to compress prohibitively costly curvature information. Moreover, we discovered several new 3PC mechanisms, such as adaptive thresholding and Bernoulli aggregation, which require reduced communication and occasional Hessian computations. Furthermore, we extend and analyze our approach to bidirectional communication compression and partial device participation setups to cater to the practical considerations of applications in federated learning. For all our methods, we derive fast condition-number-independent local linear and/or superlinear convergence rates. Finally, with extensive numerical evaluations on convex optimization problems, we illustrate that our designed schemes achieve state-of-the-art communication complexity compared to several key baselines using second-order information. 

# Code

The detailed description of each method is given in `methods.py`. `oracles.py` contains the functions that computes various characteristics of Logisitc Regression problem: local Hessians and gradients, Lipschitz constants, etc. `utils.py` consists of additional functions such as compression operators. The code is straightforward. Please, have a look at `example notebook` to see how one can run one of the implemented methods. 

# Citation

@article{newton3PC2022,

  author = {Islamov, Rustem and Qian, Xun and Hanzely, Slavomír and Safaryan, Mher and Richtárik, Peter},
  
  title = {Distributed Newton-Type Methods with Communication Compression and Bernoulli Aggregation},
  
  journal={arXiv preprint arXiv: 2206.03588}
  
  year = {2022}
}
