# Decentralized AI in Healthcare (DecAIHealth)

The project _Decentralized AI in Healthcare_ (DecAIHealth) is a project as part of the strategic program [Decentralized AI, AI Sweden](https://www.ai.se/en/projects-9/decentralized-ai). The project includes the two partners _Region Halland_ (RH) and _Västra Götalandsregionen/Sahlgrenska University Hospital_ (VGR/SU), and is coordinated by _AI Sweden_.

The overarching purpose of this project is to evaluate the possibilities for jointly training and exchanging machine learning models between Swedish healthcare regions/hospitals. For this purpose, methods will be used for decentralized training of joint machine learning models between both health regions, for example through _federated learning_.

The project includes three main phases (initially two phases, but now extended to also include an initial **Phase 0**):

* **Phase 0** establishes the project's technical feasibility through a “proof-of-concept” that the network communication is working.

* **Phase 1** verifies that decentralized machine learning models can be jointly trained based on publicly available healthcare datasets, such as the tabular dataset MIMIC-IV and the image dataset SIIM-ISIC. 

* **Phase 2** inititates by a mutal agreement upon clinical dataset and beneficial machine learning models, followed by decentralized training and validation of those models based on (both regions' own) clinical healthcare data.

<br />

### Time Plan
The project will last until the end of 2022, and a tentative time plan for the project can be found below. However, it should be noted that this time plan might be subject to changes (in mutual agreement between all the partners within the project). In addition, this time plan will also be updated to reflect the progress of the project (by indicating completed and remaining tasks).

| Date&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  | Decription    | Completed  | Required   | 
| :----------- | :------------- | :--------: | :--------: |
| 2022-04-??   | _SU_: "Dummy" server exposed externally through a fixed IP address and network port.  | &cross; | &check; |
| 2022-04-??   | **Phase 0 completed:** _RH_ verifies that an arbitrary client is able to communicate with the server at _SU_.  | &cross; | &check; |
| 2022-04-15   | _Flower_ framework installed at both _RH_ and _SU_. Initial tests to verify that models can be jointly trained and exchanged between both health regions. _Script files for initial tests\.\.\._ <br /> _RH_: `<TBA>` <br /> _SU_: `<TBA>` | &cross; | &check; |
| 2022-04-29   | Decentralized model jointly trained based on public tabular dataset (e.g., MIMIC-V). <br /> _Script files for training and validation\.\.\._ <br /> _RH_: `<TBA>` <br /> _SU_: `<TBA>` | &cross; | &check; |
| 2022-05-13   | Decentralized model jointly trained based on public image dataset (e.g., SIIM-ISIC). <br /> _Script files for training and validation\.\.\._ <br /> _RH_: `<TBA>` <br /> _SU_: `<TBA>` | &cross; | &cross; |
| 2022-05-27   | **Phase 1 completed:** test report, based on validation of jointly trained decentralized models, added to this repository.  | &cross; | &check; |

<br />

## What is Federated Learning?

![A conceptual illustration of the training cycle in federated learning](./conceptual.png)

<br />

## Federated Learning with Flower

The [Flower](https://flower.dev/) framework is a user-friendly framework designed for implementing and traning machinhe learning models in federated settings [2]. Flower is an open-source framework developed as a collaboration between the academic partners _CaMLSys Lab_, _University of Oxford_, _University of Cambridge_, and _University College London_, as well as the industrial partner _Adap_. This framework has been developed according to fundamental key characteristics required for a federated framework, such as _scalability_, _usability_, and _independency_ regarding operating systems and hardware platforms. However, the Flower framework is more than just a federated framework as it can be regarded as _" ...a unified approach to federated learning, analytics, and evaluation."_

### Installation 

An essential prerequisite for the Flower framework is a basic installation of `python` (Python 3.6 or higher version). The instructions below further assumes an installation of the `pip3` package installer for Python 3.x. To install latest stable version of Flowert (i.e., latest stable release found on PyPI):

 ```pip3 install flwr ```

  ...or to install latest unstable releases (nightly build):

 ```pip3 install flwr-nightly ```
 
 The Flower framework is also agnostic towards which machine learning framework that is used in the background (e.g., [PyTorch](https://pytorch.org/) or [TensorFlow](https://www.tensorflow.org/)). For this project, we will, however, use the PyTorch framework for training the models locally. To install the latest stable releases of PyTorch with GPU compute support of CUDA 10.2:
 
 ```pip3 install torch torchvision torchaudio```

  ...or to install the latest stable releases with GPU compute support of CUDA 11.3:
    
```pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu113```

  ...or to install the latest stable releases with CPU compute support only:
  
```pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu```

<br />  

## References

**[1]** B. McMahan, et al. _Communication-efficient learning of deep networks from decentralized data._ In Artificial intelligence and statistics, pp. 1273–1282, PMLR, 2017.

**[2]** D. J. Beutel, et al. _Flower: A friendly federated learning research framework,_ 2021.
