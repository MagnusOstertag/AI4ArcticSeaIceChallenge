# Description of the challenge

- three sea ice parameters: sea ice concentration, stage-of-development and floe size
- 493 training and 20 test (without label data) data files
- evaluation:

Sea ice parameter | Metric | Weight in total score
---|---|---
Sea ice concentration | R2 | 2/5
Stage of development | F1 | 2/5
Floe size | F1 | 1/5

# Observations on the data provided

See data manual in literature.
Also see the [introduction.ipynb](https://github.com/MagnusOstertag/AutoICE/blob/main/introduction.ipynb)

- Sentinel-1 active microwave (SAR) data and corresponding Microwave Radiometer (MWR) data from the AMSR2 satellite sensor
- SAR data has ambiguities, it has a high spatial resolution
  - The incidence angle of the SAR sensor affects the amount of radar backscatter in the image cross-section and thus this variable is included in the netCDF to enable modeling of this radiometric variation:
- MWR data has good contrast between open water and ice
- auxiliary data such as numerical weather prediction model data
- cross modal image learning or something?

# ideas

- two different models, one for the open sea and one for land areas
- problem with the wind and rain areas, can we know when they occur from the meterology
- can we learn the arctic currents?
- can we start from yesterdays model?
- seasonality and climate change should be a large factor, do we account for them?
- cross-modal transformers, visual question answering kind of
- use attention! If we know what is important we can approximate the attention with a fixed pattern

## chatGPT

"Attention U-Net: Learning Where to Look for the Pancreas" (https://arxiv.org/abs/1804.03999)
In this paper, the authors propose an Attention U-Net architecture for medical image segmentation. The model uses attention mechanisms to selectively focus on important regions of the input and improve the spatial resolution of the output.

"Deep Learning for Medical Image Segmentation using Memory-Attention U-Net" (https://arxiv.org/abs/1901.08302)
This paper presents a memory-attention U-Net architecture for medical image segmentation. The model uses attention mechanisms to focus on relevant regions of the input and uses an external memory module to store information about the spatial context of the input.

"Multi-Scale Attention U-Net for Medical Image Segmentation" (https://arxiv.org/abs/1907.05054)
In this paper, the authors propose a multi-scale Attention U-Net architecture for medical image segmentation. The model uses attention mechanisms at multiple scales to focus on relevant regions of the input and improve the spatial resolution of the output.

"Attention U-Net: Adaptive Attention via Multimodal Context Learning for Medical Image Segmentation" (https://arxiv.org/abs/2006.13848)
This paper presents an Attention U-Net architecture for medical image segmentation that uses attention mechanisms to selectively focus on relevant regions of the input. The model uses multimodal context learning to adaptively adjust the attention weights based on the input.
- use memory, somehow it should know how the weather was before, what was before in this field
"Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection" (https://arxiv.org/abs/1802.03903)
In this paper, the authors propose a memory-augmented U-Net architecture for unsupervised anomaly detection. The model uses an external memory module to store normal patterns and compares the input to the stored patterns to identify anomalies.

"Deep Learning for Medical Image Segmentation using Memory-Attention U-Net" (https://arxiv.org/abs/1901.08302)
This paper presents a memory-attention U-Net architecture for medical image segmentation. The model uses an external memory module to store information about the spatial context of the input and uses this information to guide the prediction process.

"Efficient Convolutional Neural Networks with Adaptive Inference Graphs" (https://arxiv.org/abs/2006.12435)
In this paper, the authors propose a U-Net architecture with an adaptive inference graph that uses a memory module to store and retrieve information about the input. The model can dynamically adjust the depth of the network based on the complexity of the input, leading to improved efficiency and performance.

"Memory-Efficient U-Net for Brain Tumor Segmentation" (https://arxiv.org/abs/2006.11148)
This paper presents a memory-efficient U-Net architecture for brain tumor segmentation. The model uses a memory module to store and retrieve information about the input and uses this information to guide the prediction process, leading to improved performance and reduced memory requirements.
- pretraining, you patch parts of your input and let the NN reconstruct it
  - Here are a few examples of pretraining approaches for U-Net architectures:
"Transfer learning for image segmentation with U-Net" (https://arxiv.org/abs/1707.06957)
In this paper, the authors demonstrate the effectiveness of transfer learning for U-Net architectures. They show that pretraining the U-Net on a large dataset and fine-tuning it on the target task leads to improved performance.

"Pretraining and Fine-Tuning U-Nets with Large Scale Annotated Medical Images" (https://arxiv.org/abs/2002.11972)
This paper presents a pretraining approach for U-Net architectures using a large annotated medical image dataset. The authors show that pretraining the U-Net on this dataset and fine-tuning it on the target task leads to improved performance.

"Unsupervised Pretraining for Medical Image Segmentation using Self-Supervised Learning" (https://arxiv.org/abs/2012.05594)
In this paper, the authors propose an unsupervised pretraining approach for U-Net architectures using self-supervised learning. They show that pretraining the U-Net on a large dataset using self-supervised learning and fine-tuning it on the target task leads to improved performance.

"Transfer learning for medical image segmentation using U-Net with pretrained encoder" (https://arxiv.org/abs/2008.06729)
This paper presents a transfer learning approach for U-Net architectures using a pretrained encoder. The authors show that pretraining the encoder on a large dataset and fine-tuning it on the target task leads to improved performance.

- There are several ways to improve U-Net architectures:

Increasing the depth of the network: Adding more layers to the U-Net architecture can help improve its ability to learn complex features and make more accurate predictions.

Adding skip connections: Skip connections, also known as residual connections, allow the network to bypass one or more layers and directly propagate information from the input to the output. This can help the network learn more efficiently and improve performance.

Using different types of layers: Instead of using only convolutional layers, you can try using different types of layers such as transposed convolutional layers or dilated convolutional layers. These types of layers can help improve the spatial resolution of the output and make the network more robust to small changes in the input.

Using different types of data augmentation: Data augmentation is a technique used to artificially increase the size of the training dataset by creating new samples from the existing ones. This can help the network generalize better and improve its performance on unseen data.

Using different optimization algorithms: You can try using different optimization algorithms such as Adam, RMSprop, or SGD to train the network. Each algorithm has its own set of hyperparameters that can be tuned to improve the network's performance.

Using different loss functions: Different loss functions can be used to measure the difference between the predicted output and the ground truth. You can try using different loss functions such as cross-entropy loss, mean squared error loss, or mean absolute error loss to see which one gives the best performance.

- training on similar tasks on as with language.

# human analysts

The conditions are described by multiple parameters and follow the World Meteorological Organization (WMO) code for sea ice—Sea Ice GeoReferenced Information and Data (SIGRID3). The primary descriptive factor is SIC—a metric from 0% to 100%, indicating the ratio of sea ice to open water, where 0% is ice-free open water and 100% is fully covered sea ice. The ice concentration mapping is created through a creative process of individual interpretation steered by common guidelines with no associated uncertainty. However, studies have suggested that ice analysts assign concentrations that vary on average 20% and up to 60% discrepancies [18]. Intermediate SICs (10%–90%) are particularly difficult to assess. The regions near the edge of the sea ice cover—called the marginal ice zone—receive more attention because it is the most important area for maritime operations. In comparison, inner ice areas with low maritime activity receive less attention. Despite these uncertainties, we treat each pixel as equally valid.  [AI4SeaIce: Toward Solving Ambiguous SAR extures in Convolutional Neural Networks for Automatic Sea Ice Concentration Charting](https://ieeexplore.ieee.org/document/9705586)

# Applying concepts from the lecture

## model formalization



## frankenstein models


### pre-trained models?

# what is a good result?

- U-Net CNN on SAR for sea ice prediction with R2-score of `86.34%` in [AI4SeaIce: Toward Solving Ambiguous SAR extures in Convolutional Neural Networks for Automatic Sea Ice Concentration Charting](https://ieeexplore.ieee.org/document/9705586)
- CNN on SAR, AMSR2 for sea ice prediction with R2 score of `89%` in [Architecture for Sentinel-1 and AMSR2 Data Fusion](https://ieeexplore.ieee.org/document/9133205)

# Literature

- [AI4SeaIce: Toward Solving Ambiguous SAR extures in Convolutional Neural Networks for Automatic Sea Ice Concentration Charting](https://ieeexplore.ieee.org/document/9705586)
  - based on SAR it is a hard task for CNNs
    - the receptive field is important: 3068px
    - noise phenomenon is present in the Sentinel-1 ESA Instrument Processing Facility (IPF) v2.9 SAR data, particularly in subswath transitions, visible as long vertical lines and grained particles resembling small sea ice floes
  - U-Net CNN architecture plus symetrical blocks of convolutional, pooling and upsampling layers in the encoder and decoder of the U-Net
  - training on noise-corrected SAR data - NERSC
  - U-net code available at [github](https://github.com/astokholm/AI4SeaIce.git.)
  - There may be a trade-off between the level of detail and homogeneity—a larger receptive field increases the homogeneity of SIC predictions, but it also appears to reduce the level of detail in predictions
  - landfast ice predictions are still troublesome for the model
- [Architecture for Sentinel-1 and AMSR2 Data Fusion](https://ieeexplore.ieee.org/document/9133205)
  - image segmentation like CNN
  - fusion of data from SAR and AMSR2
- [A labelled ocean SAR imagery dataset of ten geophysical phenomena from Sentinel-1 wave mode](https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/gdj3.73) and others
- [Construction of a climate data record of sea surface temperature from passive microwave measurements](https://www.sciencedirect.com/science/article/abs/pii/S0034425719305048)
  - measure the sea temperature

## ToRead

- [Classification of Sea Ice Types in Sentinel-1 SAR Data Using Convolutional Neural Networks](https://www.mdpi.com/2072-4292/12/13/2165)
- [Classification of sea ice types in Sentinel-1 synthetic aperture radar images](https://tc.copernicus.org/articles/14/2629/2020/)
- [Satellite-derived maps of Arctic and Antarctic sea ice motion: 1988 to 1994](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/97GL00755)
- [Seasonal Arctic sea ice forecasting with probabilistic deep learning](https://www.nature.com/articles/s41467-021-25257-4?tpcc=nleyeonai)
- [Proof of Concept for Sea Ice Stage of Development Classification Using Deep Learning ](https://www.mdpi.com/2072-4292/12/15/2486)
- [MACHINE LEARNING FOR SEA ICE MONITORING FROM SATELLITES](https://noa.gwlb.de/servlets/MCRFileNodeServlet/cop_derivate_00040864/isprs-archives-XLII-2-W16-83-2019.pdf)
- [On the influence of model physics on simulations of Arctic and Antarctic sea ice](https://tc.copernicus.org/articles/5/687/2011/)
- [An enhancement of the NASA Team sea ice algorithm](https://ieeexplore.ieee.org/abstract/document/843033)
- [Chapter 1 The Role of Sea Ice in Arctic and Antarctic Polynyas](https://www.sciencedirect.com/science/article/abs/pii/S0422989406740016)
- [A new tracking algorithm for sea ice age distribution estimation](https://tc.copernicus.org/articles/12/2073/2018/tc-12-2073-2018.pdf)
- [Automatic satellite-based ice charting using AI](https://orbit.dtu.dk/en/publications/automatic-satellite-based-ice-charting-using-ai)
- [Simulated geophysical noise in the Copernicus Imaging Microwave Radiometer (CIMR) ice concentration estimates over snow covered sea ice](https://digital.csic.es/handle/10261/205017)

# Tools

## MLflow

[documentation](https://www.mlflow.org/docs/latest/tracking.html)

- mlflow tracking: autologging for classic libraries, UI to directly visualize tracked metrics
- mlflow projects: packaging code to reuse or chain together
- mlflow models with flavors
- mlflow model registry to manage the full lifecycle of a mlflow model


# ToDo

- [ ] read the data manual
- [x] read the abstracts of papers by the authors of the dataset
- [ ] download some data and try out the `ipynb`
- [ ] read the further papers
- [ ] formalize the problem
- [ ] have for learning pipelines

- [ ] how does requesting the computing resources work? I do not know how to log-in the website, did @schererant get information in a mail?
- [x] ask Prof. Lombardi for
  - tips
  - what kind of DL models could be useful?
  - how to tackle the challenge?
  - when are we good? Do we have to apply the concepts of the lecture
  - how much time are we expected to spend on the task

# Tips from prof

- graphCNN, learn transitions
- constraints
- do not overoptimise hyperparams

# possible further research

- SAR noise correction is done according to state-of-the-art techniques by Korosov et al. (doi: 10.1109/TGRS.2021.3131036). What are the known limits?
- More information about the Sentinel-1
sensor and data can be found at https://sentinel.esa.int/web/sentinel/missions/sentinel-1.
