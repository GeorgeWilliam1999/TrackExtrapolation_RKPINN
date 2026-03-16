# Reference Papers for Neural Track Extrapolation

This directory contains references to all academic papers cited in this project. The papers are organized by topic and include links for download where available.

## How to Access Papers

Most papers are available through:
- **arXiv**: Free preprint server (arxiv.org)
- **DOI links**: Direct journal access (may require institutional subscription)
- **CERN Document Server**: For LHCb/CERN publications (cds.cern.ch)

---

## 1. LHCb and Particle Physics

### LHC Machine
- **Authors**: L. Evans and P. Bryant (eds.)
- **Title**: "LHC Machine"
- **Journal**: JINST 3, S08001 (2008)
- **DOI**: [10.1088/1748-0221/3/08/S08001](https://doi.org/10.1088/1748-0221/3/08/S08001)
- **Description**: Comprehensive description of the Large Hadron Collider design and performance

### LHCb Detector
- **Authors**: LHCb Collaboration
- **Title**: "The LHCb Detector at the LHC"
- **Journal**: JINST 3, S08005 (2008)
- **DOI**: [10.1088/1748-0221/3/08/S08005](https://doi.org/10.1088/1748-0221/3/08/S08005)
- **Description**: Original LHCb detector technical design and specifications

### LHCb Upgrade II Framework TDR
- **Authors**: LHCb Collaboration
- **Title**: "Framework TDR for the LHCb Upgrade II: Opportunities in flavour physics, and beyond, in the HL-LHC era"
- **Report**: CERN-LHCC-2021-012, LHCb-TDR-023 (2021)
- **DOI**: [10.17181/CERN.NTVH.Q21W](https://doi.org/10.17181/CERN.NTVH.Q21W)
- **CDS Link**: [https://cds.cern.ch/record/2776420](https://cds.cern.ch/record/2776420)
- **Description**: Technical design for LHCb Upgrade II (2033+), key source for luminosity and trigger requirements

### Allen GPU Trigger
- **Authors**: R. Aaij et al.
- **Title**: "Allen: A high-level trigger on GPUs for LHCb"
- **Journal**: Comput. Softw. Big Sci. 4, 7 (2020)
- **DOI**: [10.1007/s41781-020-00039-7](https://doi.org/10.1007/s41781-020-00039-7)
- **arXiv**: [1906.08875](https://arxiv.org/abs/1906.08875)
- **Description**: Description of LHCb's GPU-based high-level trigger system

---

## 2. Physics-Informed Neural Networks (PINNs)

### Original PINN Paper (Foundational - 12,000+ citations)
- **Authors**: M. Raissi, P. Perdikaris, and G.E. Karniadakis
- **Title**: "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations"
- **Journal**: J. Comput. Phys. 378, 686-707 (2019)
- **DOI**: [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)
- **Open Access**: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- **Description**: **Seminal paper introducing PINNs**. Establishes the framework for encoding physical laws into neural network training through loss functions.

### PINN Review (Nature Reviews Physics)
- **Authors**: G.E. Karniadakis et al.
- **Title**: "Physics-informed machine learning"
- **Journal**: Nature Rev. Phys. 3, 422-440 (2021)
- **DOI**: [10.1038/s42254-021-00314-5](https://doi.org/10.1038/s42254-021-00314-5)
- **Description**: Comprehensive review of physics-informed ML methods, benefits, and applications

### Neural Ordinary Differential Equations
- **Authors**: R.T.Q. Chen, Y. Rubanova, J. Bettencourt, and D. Duvenaud
- **Title**: "Neural Ordinary Differential Equations"
- **Conference**: NeurIPS 2018 (Best Paper Award)
- **arXiv**: [1806.07366](https://arxiv.org/abs/1806.07366)
- **Description**: **Key paper for RK-PINN motivation**. Shows connection between ResNets and ODE solvers, enabling continuous-depth networks.

### Multi-scale PINNs
- **Authors**: S. Wang, H. Wang, and P. Perdikaris
- **Title**: "On the eigenvector bias of Fourier feature networks: From regression to solving multi-scale PDEs with physics-informed neural networks"
- **Journal**: Comput. Methods Appl. Mech. Eng. 384, 113938 (2021)
- **DOI**: [10.1016/j.cma.2021.113938](https://doi.org/10.1016/j.cma.2021.113938)
- **arXiv**: [2012.10047](https://arxiv.org/abs/2012.10047)
- **Description**: Addresses multi-scale problems in PINNs using Fourier features

### Fourier Neural Operator
- **Authors**: Z. Li et al.
- **Title**: "Fourier Neural Operator for Parametric Partial Differential Equations"
- **Conference**: ICLR 2021
- **arXiv**: [2010.08895](https://arxiv.org/abs/2010.08895)
- **Description**: Neural operator approach for solving PDEs, alternative to PINNs

---

## 3. Neural Network Architectures

### Deep Residual Learning (ResNet)
- **Authors**: K. He, X. Zhang, S. Ren, and J. Sun
- **Title**: "Deep Residual Learning for Image Recognition"
- **Conference**: CVPR 2016
- **DOI**: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)
- **arXiv**: [1512.03385](https://arxiv.org/abs/1512.03385)
- **Description**: **Foundational paper for skip connections**. Introduces residual learning framework that enables training of very deep networks. Key motivation for our Residual MLP architecture.

### DenseNet
- **Authors**: G. Huang, Z. Liu, L. van der Maaten, and K.Q. Weinberger
- **Title**: "Densely Connected Convolutional Networks"
- **Conference**: CVPR 2017
- **DOI**: [10.1109/CVPR.2017.243](https://doi.org/10.1109/CVPR.2017.243)
- **arXiv**: [1608.06993](https://arxiv.org/abs/1608.06993)
- **Description**: Dense connections between layers, related to multi-head architecture design

### Wide Residual Networks
- **Authors**: S. Zagoruyko and N. Komodakis
- **Title**: "Wide Residual Networks"
- **Conference**: BMVC 2016
- **arXiv**: [1605.07146](https://arxiv.org/abs/1605.07146)
- **Description**: Explores width vs depth trade-offs, motivates our wide-shallow architectures

### Universal Approximation Theorem
- **Authors**: K. Hornik, M. Stinchcombe, and H. White
- **Title**: "Multilayer feedforward networks are universal approximators"
- **Journal**: Neural Networks 2, 359-366 (1989)
- **DOI**: [10.1016/0893-6080(89)90020-8](https://doi.org/10.1016/0893-6080(89)90020-8)
- **Description**: **Theoretical foundation for MLPs**. Proves that neural networks can approximate any continuous function.

---

## 4. Activation Functions

### SiLU/Swish Activation
- **Authors**: P. Ramachandran, B. Zoph, and Q.V. Le
- **Title**: "Searching for Activation Functions"
- **arXiv**: [1710.05941](https://arxiv.org/abs/1710.05941)
- **Description**: **Key paper for our activation function choice**. Uses automated search to discover Swish/SiLU: f(x) = x·sigmoid(βx). Shows consistent improvements over ReLU on ImageNet (+0.6-0.9%).

### GELU Activation
- **Authors**: D. Hendrycks and K. Gimpel
- **Title**: "Gaussian Error Linear Units (GELUs)"
- **arXiv**: [1606.08415](https://arxiv.org/abs/1606.08415)
- **Description**: Alternative smooth activation function: GELU(x) = x·Φ(x). Used in BERT and GPT models.

---

## 5. Optimization

### AdamW Optimizer
- **Authors**: I. Loshchilov and F. Hutter
- **Title**: "Decoupled Weight Decay Regularization"
- **Conference**: ICLR 2019
- **arXiv**: [1711.05101](https://arxiv.org/abs/1711.05101)
- **Description**: **Optimizer used in training**. Fixes weight decay implementation in Adam, showing improved generalization.

---

## 6. Numerical Methods and Physics

### Hairer ODE Book
- **Authors**: E. Hairer, S.P. Nørsett, and G. Wanner
- **Title**: "Solving Ordinary Differential Equations I: Nonstiff Problems"
- **Publisher**: Springer (1993)
- **ISBN**: 978-3540566700
- **Description**: Authoritative reference for Runge-Kutta methods. Chapter II covers RK4 and embedded methods.

### Jackson Classical Electrodynamics
- **Authors**: J.D. Jackson
- **Title**: "Classical Electrodynamics" (3rd ed.)
- **Publisher**: Wiley (1998)
- **ISBN**: 978-0471309321
- **Description**: Standard reference for Lorentz force and particle motion in electromagnetic fields.

### GEANT4 Physics
- **Authors**: GEANT4 Collaboration
- **Title**: "Geant4—a simulation toolkit"
- **Journal**: Nucl. Instrum. Methods A 506, 250-303 (2003)
- **DOI**: [10.1016/S0168-9002(03)01368-8](https://doi.org/10.1016/S0168-9002(03)01368-8)
- **Description**: GEANT4 particle transport simulation, including magnetic field propagation algorithms.

---

## 7. Deep Learning Frameworks

### PyTorch
- **Authors**: A. Paszke et al.
- **Title**: "PyTorch: An Imperative Style, High-Performance Deep Learning Library"
- **Conference**: NeurIPS 2019
- **arXiv**: [1912.01703](https://arxiv.org/abs/1912.01703)
- **Description**: Deep learning framework used for all model training in this project.

### Deep Learning Book
- **Authors**: I. Goodfellow, Y. Bengio, and A. Courville
- **Title**: "Deep Learning"
- **Publisher**: MIT Press (2016)
- **ISBN**: 978-0262035613
- **Online**: [www.deeplearningbook.org](https://www.deeplearningbook.org/)
- **Description**: Comprehensive textbook covering neural network fundamentals, optimization, and regularization.

---

## 8. Ensemble Methods

### Bagging Predictors
- **Authors**: L. Breiman
- **Title**: "Bagging predictors"
- **Journal**: Machine Learning 24, 123-140 (1996)
- **DOI**: [10.1007/BF00058655](https://doi.org/10.1007/BF00058655)
- **Description**: Foundational ensemble method. Relevant to understanding multi-head prediction architecture.

---

## Citation Summary

When citing this work, please include references to the foundational papers:

1. **For PINNs**: Raissi et al. (2019) - J. Comput. Phys.
2. **For ResNets/Skip Connections**: He et al. (2016) - CVPR
3. **For Neural ODEs**: Chen et al. (2018) - NeurIPS
4. **For SiLU/Swish**: Ramachandran et al. (2017) - arXiv
5. **For AdamW**: Loshchilov & Hutter (2019) - ICLR
6. **For LHCb**: LHCb Collaboration (2008, 2021) - JINST, CERN

---

## Download Instructions

### arXiv Papers
```bash
# Example: Download PINN paper
wget https://arxiv.org/pdf/1806.07366.pdf -O neural_ode.pdf

# Example: Download ResNet paper  
wget https://arxiv.org/pdf/1512.03385.pdf -O resnet.pdf

# Example: Download Swish paper
wget https://arxiv.org/pdf/1710.05941.pdf -O swish.pdf
```

### CERN Documents
Access via CERN Document Server: https://cds.cern.ch/

---

**Last Updated**: January 2026
**Maintainer**: George William Scriven
