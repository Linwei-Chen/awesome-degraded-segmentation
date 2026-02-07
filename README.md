# 🔬 A Survey on Degraded Image Segmentation

<div align="center">

![Taxonomy](assets/images/taxonomy_overview.png)

**A comprehensive survey on robust image segmentation under various degradation conditions**

[![GitHub Stars](https://img.shields.io/github/stars/Linwei-Chen/awesome-degraded-segmentation?style=social)](https://github.com/Linwei-Chen/awesome-degraded-segmentation)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-CJE%202025-green.svg)](https://cje.ejournal.org.cn/)

[📖 Paper](#abstract) | [🗂️ Paper List](#paper-list) | [🌐 Project Page](https://linwei-chen.github.io/awesome-degraded-segmentation/)

</div>

---

## Abstract

Image segmentation is a fundamental task in computer vision with wide-ranging applications. While deep learning models have achieved remarkable success under ideal conditions, their performance often degrades catastrophically when faced with real-world image corruptions. These corruptions span several key categories, including adverse weather (*e.g.*, fog, rain, snow), challenging light (*e.g.*, nighttime, low-light), digital artifacts from processing (*e.g.*, compression, color jitter), various forms of blur (*e.g.*, motion, defocus), and pervasive noise (*e.g.*, sensor noise, speckle). This degradation poses a critical challenge, particularly in safety-sensitive domains like autonomous driving, medical imaging, and remote sensing. This survey provides a comprehensive and structured overview of the field of degraded image segmentation, offering a clearer understanding of overarching approaches compared to prior surveys that often focus solely on degradation types.  We establish a detailed taxonomy of common image degradations impacting segmentation tasks. We review a wide array of datasets and benchmarks designed for evaluating robustness. Furthermore, we systematically analyze state-of-the-art methodologies, categorized by their core technical strategies: domain adaptation and generalization, joint restoration-segmentation techniques, and multi-modal fusion. Finally, we identify critical open challenges and discuss promising future research directions, aiming to synthesize current knowledge and stimulate targeted advancements towards more generalizable, adaptable, and reliable segmentation systems. Detailed paper summaries are available at [https://github.com/Linwei-Chen/awesome-degraded-segmentation](https://github.com/Linwei-Chen/awesome-degraded-segmentation).

## 📊 Degradation Examples

![Degradation Examples](assets/images/degraded_image.png)

*Examples of various image degradation types: weather, light, digital, blur, and noise.*

---

## 🌟 Highlights

- **42 papers with open-source code** are highlighted and prioritized
- Comprehensive coverage of **5 degradation categories**: Weather, Light, Digital, Blur, Noise
- **3 main methodological strategies**: Domain Adaptation/Generalization, Joint Restoration-Segmentation, Multi-modal Fusion

---

## Paper List

> 📌 **Note**: Papers with available code implementations are marked with ⭐ and listed first in each category.


## 1. Domain Adaptation & Generalization (DA/DG)

### 1.2 Feature Alignment

|  | Method | Title | Paper | Code | Venue | Year | BibTeX |
|:--:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| ⭐ | CISS | **Condition-Invariant Semantic Segmentation** | [Paper](Condition-Invariant_Semantic_Segmentation.pdf) | [Code](https://github.com/SysCV/CISS) | PAMI | 2025 | [BibTeX](assets/bibtex/sakaridis2025condition.txt) |
| ⭐ | ICDA | **ICDA: Illumination-Coupled Domain Adaptation Framework for Unsupervised Nighttime Semantic Segmentation** | [Paper](ICDA%20Illumination-Coupled%20Domain%20Adaptation%20Framework%20for%20Unsupervised%20Nighttime%20Semantic%20Segmentation.pdf) | [Code](https://github.com/chenghaoDong666/ICDA) | IJCAI | 2023 | [BibTeX](assets/bibtex/dong2023icda.txt) |
| ⭐ | - | **Degraded Image Semantic Segmentation Using Intra-image and Inter-image Contrastive Learning** | [Paper](Degraded_Image_Semantic_Segmentation_Using_Intra-image_and_Inter-image_Contrastive_Learning.pdf) | [Code](https://github.com/cocolord/degraded_image_seg) | CAC | 2023 | [BibTeX](assets/bibtex/dong2023degraded.txt) |
| ⭐ | CCDistill | **Cross-Domain Correlation Distillation for Unsupervised Domain Adaptation in Nighttime Semantic Segmentation** | [Paper](Gao_Cross-Domain_Correlation_Distillation_for_Unsupervised_Domain_Adaptation_in_Nighttime_Semantic_CVPR_2022_paper.pdf) | [Code](https://github.com/ghuan99/CCDistill) | CVPR | 2022 | [BibTeX](assets/bibtex/gao2022cross.txt) |
|  | - | **Learning Texture Invariant Representation for Domain Adaptation of Semantic Segmentation** | [Paper](Kim_Learning_Texture_Invariant_Representation_for_Domain_Adaptation_of_Semantic_Segmentation_CVPR_2020_paper.pdf) | - | CVPR | 20020 | [BibTeX](assets/bibtex/kim2020learning.txt) |
|  | CIADA | **Computational Imaging for Machine Perception: Transferring Semantic Segmentation Beyond Aberrations** | [Paper](Computational_Imaging_for_Machine_Perception_Transferring_Semantic_Segmentation_Beyond_Aberrations.pdf) | - | TCI | 2024 | [BibTeX](assets/bibtex/jiang2024computational.txt) |
|  | - | **Learning intra-domain style-invariant representation for unsupervised domain adaptation of semantic segmentation** | [Paper](Learning%20intra-domain%20style-invariant%20representation%20for%20unsupervised%20domain%20adaptation%20of%20semantic%20segmentation.pdf) | - | PR | 2022 | [BibTeX](assets/bibtex/li2022learning.txt) |
|  | - | **Cluster Alignment With Target Knowledge Mining for Unsupervised Domain Adaptation Semantic Segmentation** | [Paper](Cluster_Alignment_With_Target_Knowledge_Mining_for_Unsupervised_Domain_Adaptation_Semantic_Segmentation.pdf) | - | TIP | 2022 | [BibTeX](assets/bibtex/wang2022cluster.txt) |
|  | - | **Semantic Nighttime Image Segmentation Via Illumination and Position Aware Domain Adaptation** | [Paper](Semantic_Nighttime_Image_Segmentation_Via_Illumination_and_Position_Aware_Domain_Adaptation.pdf) | - | ICIP | 2021 | [BibTeX](assets/bibtex/peng2021semantic.txt) |
|  | RSSN | **Nighttime Road Scene Parsing by Unsupervised Domain Adaptation** | [Paper](Nighttime_Road_Scene_Parsing_by_Unsupervised_Domain_Adaptation.pdf) | - | TITS | 2020 | [BibTeX](assets/bibtex/song2020nighttime.txt) |


## 2. Joint Restoration & Segmentation

### 2.2 Deraining + Segmentation

|  | Method | Title | Paper | Code | Venue | Year | BibTeX |
|:--:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| ⭐ | DRNet | **DRNet: Learning a dynamic recursion network for chaotic rain streak removal** | [Paper](DRNet%20Learning%20a%20dynamic%20recursion%20network%20for%20chaotic%20rain%20streak%20removal.pdf) | [Code](https://github.com/Jzy2017/DRNet) | PR | 2025 | [BibTeX](assets/bibtex/jiang2025drnet.txt) |
| ⭐ | RCDNet | **RCDNet: An interpretable rain convolutional dictionary network for single image deraining** | [Paper](RCDNet_An_Interpretable_Rain_Convolutional_Dictionary_Network_for_Single_Image_Deraining.pdf) | [Code](https://github.com/hongwang01/DRCDNet) | TNNLS | 2023 | [BibTeX](assets/bibtex/wang2023rcdnet.txt) |
| ⭐ | EPRRNet/PRRNets | **Beyond Monocular Deraining: Parallel Stereo Deraining Network Via Semantic Prior** | [Paper](Beyond%20Monocular%20Deraining%20Parallel%20Stereo%20Deraining%20Network%20Via%20Semantic%20Prior.pdf) | [Code](https://github.com/HDCVLab/Stereo-Image-Deraining) | IJCV | 2022 | [BibTeX](assets/bibtex/zhang2022beyond.txt) |
| ⭐ | - | **I Can See Clearly Now : Image Restoration via De-Raining** | [Paper](I_Can_See_Clearly_Now_Image_Restoration_via_De-Raining.pdf) | [Code](https://ciumonk.github.io/RobotCar-rainy/) | ICRA | 2019 | [BibTeX](assets/bibtex/porav2019can.txt) |
|  | - | **Learning A Rain-Invariant Network For Instance Segmentation In The Rain** | [Paper](Learning_A_Rain-Invariant_Network_For_Instance_Segmentation_In_The_Rain.pdf) | - | ICIP | 2024 | [BibTeX](assets/bibtex/chen2024learning.txt) |
|  | - | **Real Rainy Scene Analysis: A Dual-Module Benchmark for Image Deraining and Segmentation** | [Paper](Real_Rainy_Scene_Analysis_A_Dual-Module_Benchmark_for_Image_Deraining_and_Segmentation.pdf) | - | ICMEW | 2023 | [BibTeX](assets/bibtex/zhao2023real.txt) |
|  | - | **Improved Sea-Ice Identification Using Semantic Segmentation With Raindrop Removal** | [Paper](Improved_Sea-Ice_Identification_Using_Semantic_Segmentation_With_Raindrop_Removal.pdf) | - | IEEE | 2022 | [BibTeX](assets/bibtex/alsharay2022improved.txt) |


## 3. Multi-Modal Fusion

### 3.2 RGB + LiDAR/Depth Fusion

|  | Method | Title | Paper | Code | Venue | Year | BibTeX |
|:--:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| ⭐ | LLE-Seg | **Low-Light Enhancement and Global-Local Feature Interaction for RGB-T Semantic Segmentation** | [Paper](Low-Light_Enhancement_and_Global-Local_Feature_Interaction_for_RGB-T_Semantic_Segmentation.pdf) | [Code](https://github.com/Yuyu-1015/ LLE-Seg) | TIM | 2025 | [BibTeX](assets/bibtex/guo2025low.txt) |
|  | - | **Adaptive Entropy Multi-modal Fusion for Nighttime Lane Segmentation** | [Paper](Adaptive_Entropy_Multi-modal_Fusion_for_Nighttime_Lane_Segmentation.pdf) | - | TIV | 2024 | [BibTeX](assets/bibtex/zhang2024adaptive.txt) |
|  | - | **Semantic Segmentation Research of Motion Blurred Images by Event Camera** | [Paper](Semantic_Segmentation_Research_of_Motion_Blurred_Images_by_Event_Camera.pdf) | - | CVCI | 2023 | [BibTeX](assets/bibtex/liu2023semantic.txt) |
|  | GNN | **Multi-Robot Collaborative Perception With Graph Neural Networks** | [Paper](Multi-Robot_Collaborative_Perception_With_Graph_Neural_Networks.pdf) | - | IEEE | 2022 | [BibTeX](assets/bibtex/zhou2022multi.txt) |
|  | UNO | **UNO: Uncertainty-aware Noisy-Or Multimodal Fusion for Unanticipated Input Degradation** | [Paper](UNO_Uncertainty-aware_Noisy-Or_Multimodal_Fusion_for_Unanticipated_Input_Degradation.pdf) | - | ICRA | 2020 | [BibTeX](assets/bibtex/tian2020uno.txt) |

### 3.3 RGB + Event Camera Fusion

|  | Method | Title | Paper | Code | Venue | Year | BibTeX |
|:--:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| ⭐ | LF-DEST | **Incorporating degradation estimation in light field spatial super-resolution** | [Paper](/) | [Code](https://github.com/zeyuxiao1997/LF-DEST) | Computer Vision and Image Understanding | 2025 | [BibTeX](assets/bibtex/xiao2025incorporating_br_.txt) |
| ⭐ | LLE-VOS/LLE-DAVIS | **Event-assisted Low-Light Video Object Segmentation** | [Paper](Li_Event-assisted_Low-Light_Video_Object_Segmentation_CVPR_2024_paper.pdf) | [Code](https://github.com/HebeiFast/EventLowLightVOS) | CVPR | 2024 | [BibTeX](assets/bibtex/li2024event.txt) |
| ⭐ | CMDA | **CMDA: Cross-Modality Domain Adaptation for Nighttime Semantic Segmentation** | [Paper](Xia_CMDA_Cross-Modality_Domain_Adaptation_for_Nighttime_Semantic_Segmentation_ICCV_2023_paper.pdf) | [Code](https://github.com/XiaRho/CMDA) | ICCV | 2023 | [BibTeX](assets/bibtex/xia2023cmda.txt) |


## 1. Domain Adaptation & Generalization (DA/DG)

### 1.3 Feature Decomposition

|  | Method | Title | Paper | Code | Venue | Year | BibTeX |
|:--:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| ⭐ | DAF | **WHEN SEMANTIC SEGMENTATION MEETS FREQUENCY ALIASING** | [Paper](WHEN%20SEMANTIC%20SEGMENTATION%20MEETS%20FREQUENCY%20ALIASING.pdf) | [Code](https://github.com/Linwei-Chen/Seg-Aliasing) | ICLR | 2024 | [BibTeX](assets/bibtex/chen2024semantic.txt) |
|  | DDFL | **DDFL: Dual-Domain Feature Learning for nighttime semantic segmentation** | [Paper](DDFL%20Dual-Domain%20Feature%20Learning%20for%20nighttime%20semantic%20segmentation.pdf) | - | Displays | 2024 | [BibTeX](assets/bibtex/lin2024ddfl.txt) |
|  | - | **Both Style and Fog Matter: Cumulative Domain Adaptation for Semantic Foggy Scene Understanding** | [Paper](Ma_Both_Style_and_Fog_Matter_Cumulative_Domain_Adaptation_for_Semantic_CVPR_2022_paper.pdf) | - | CVPR | 2022 | [BibTeX](assets/bibtex/ma2022both.txt) |

### 1.4 Self-Training & Pseudo-Labeling

|  | Method | Title | Paper | Code | Venue | Year | BibTeX |
|:--:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| ⭐ | - | **Source-Free Online Domain Adaptive Semantic Segmentation of Satellite Images Under Image Degradation** | [Paper](Source-Free_Online_Domain_Adaptive_Semantic_Segmentation_of_Satellite_Images_Under_Image_Degradation.pdf) | [Code](https://sat-tta.github.io/) | ICASSP | 2024 | [BibTeX](assets/bibtex/niloy2024source.txt) |
| ⭐ | VBLC | **VBLC: Visibility boosting and logit-constraint learning for domain adaptive semantic segmentation under adverse conditions** | [Paper](VBLC%20Visibility%20boosting%20and%20logit-constraint%20learning%20for%20domain%20adaptive%20semantic%20segmentation%20under%20adverse%20conditions.pdf) | [Code](https://github.com/BIT-DA/VBLC) | AAAI | 2023 | [BibTeX](assets/bibtex/li2023vblc.txt) |
| ⭐ | LoopDA | **LoopDA: Constructing Self-loops to Adapt Nighttime Semantic Segmentation** | [Paper](Shen_LoopDA_Constructing_Self-Loops_To_Adapt_Nighttime_Semantic_Segmentation_WACV_2023_paper.pdf) | [Code](https://github.com/fy-vision/LoopDA) | WACV | 2023 | [BibTeX](assets/bibtex/shen2023loopda.txt) |
| ⭐ | DTBS | **Dtbs: Dual-teacher bi-directional self-training for domain adaptation in nighttime semantic segmentation** | [Paper](Dtbs%20Dual-teacher%20bi-directional%20self-training%20for%20domain%20adaptation%20in%20nighttime%20semantic%20segmentation.pdf) | [Code](https://github.com/hf618/DTBS) | ECAI | 2023 | [BibTeX](assets/bibtex/huang2023dtbs.txt) |
| ⭐ | - | **Unsupervised foggy scene understanding via self spatial-temporal label diffusion** | [Paper](Unsupervised_Foggy_Scene_Understanding_via_Self_Spatial-Temporal_Label_Diffusion.pdf) | [Code](http://people.ee.ethz.ch/∼ csakarid/SFSU_synthetic/) | TIP | 2022 | [BibTeX](assets/bibtex/liao2022unsupervised.txt) |
| ⭐ | - | **Self Pseudo Entropy Knowledge Distillation for Semi-Supervised Semantic Segmentation** | [Paper](Self_Pseudo_Entropy_Knowledge_Distillation_for_Semi-Supervised_Semantic_Segmentation.pdf) | [Code](https://github.com/xiaoqiang-lu/SPEED) | CVPR | 2022 | [BibTeX](assets/bibtex/bar2022performance.txt) |
| ⭐ | SNE-Seg | **SS-SFDA: Self-Supervised Source-Free Domain Adaptation for Road Segmentation in Hazardous Environments** | [Paper](Kothandaraman_SS-SFDA_Self-Supervised_Source-Free_Domain_Adaptation_for_Road_Segmentation_in_Hazardous_ICCVW_2021_paper.pdf) | [Code](https://gamma.umd.edu/weatherSAfE/) | ICCV | 2021 | [BibTeX](assets/bibtex/kothandaraman2021ss.txt) |
| ⭐ | Heatnet | **Heatnet: Bridging the day-night domain gap in semantic segmentation with thermal images** | [Paper](HeatNet_Bridging_the_Day-Night_Domain_Gap_in_Semantic_Segmentation_with_Thermal_Images.pdf) | [Code](http://thermal.cs.uni-freiburg.de/) | IROS | 2020 | [BibTeX](assets/bibtex/vertens2020heatnet.txt) |
|  | SDAT-Former++ | **SDAT-Former++ A Foggy Scene Semantic Segmentation Method with Stronger Domain Adaption Teacher for Remote Sensing Images** | [Paper](SDAT-Former++%20A%20Foggy%20Scene%20Semantic%20Segmentation%20Method%20with%20Stronger%20Domain%20Adaption%20Teacher%20for%20Remote%20Sensing%20Images.pdf) | - | MDPI | 2023 | [BibTeX](assets/bibtex/wang2023sdat.txt) |
|  | PLS-DAFormer | **A Two-Stage Self-Training Framework for Nighttime Semantic Segmentation** | [Paper](A_Two-Stage_Self-Training_Framework_for_Nighttime_Semantic_Segmentation.pdf) | - | YAC | 2023 | [BibTeX](assets/bibtex/yang2023two.txt) |
|  | SGDA | **SGDA: A Saliency-Guided Domain Adaptation Network for Nighttime Semantic Segmentation** | [Paper](SGDA_A_Saliency-Guided_Domain_Adaptation_Network_for_Nighttime_Semantic_Segmentation.pdf) | - | ICPS | 2023 | [BibTeX](assets/bibtex/duan2023sgda.txt) |
|  | DCL | **Dual-level Consistency Learning for Unsupervised Domain Adaptive Night-time Semantic Segmentation** | [Paper](Dual-level_Consistency_Learning_for_Unsupervised_Domain_Adaptive_Night-time_Semantic_Segmentation.pdf) | - | ICME | 2023 | [BibTeX](assets/bibtex/ding2023dual.txt) |
|  | MADA | **MADA: Multi-Level Alignment in Domain Adaptation Network for Nighttime Semantic Segmentation** | [Paper](MADA_Multi-Level_Alignment_in_Domain_Adaptation_Network_for_Nighttime_Semantic_Segmentation.pdf) | - | ICIVC | 2023 | [BibTeX](assets/bibtex/xu2023mada.txt) |
|  | HDL | **A hybrid domain learning framework for unsupervised semantic segmentation** | [Paper](A%20hybrid%20domain%20learning%20framework%20for%20unsupervised%20semantic%20segmentation.pdf) | - | Neurocomputing | 2023 | [BibTeX](assets/bibtex/zhang2023hybrid.txt) |
|  | AUGCO | **AUGCO: Augmentation Consistency-guided Self-training for Source-free Domain Adaptive Semantic Segmentation** | [Paper](AUGCO%20Augmentation%20Consistency-guided%20Self-training%20for%20Source-free%20Domain%20Adaptive%20Semantic%20Segmentation.pdf) | - | NeurIPS | 2022 | [BibTeX](assets/bibtex/prabhu2022augmentation.txt) |
|  | MTKD | **Weather-degraded image semantic segmentation with multi-task knowledge distillation** | [Paper](Weather-degraded%20image%20semantic%20segmentation%20with%20multi-task%20knowledge%20distillation.pdf) | - | IV | 2022 | [BibTeX](assets/bibtex/li2022weather.txt) |
|  | RanPaste | **RanPaste: Paste Consistency and Pseudo Label for Semisupervised Remote Sensing Image Semantic Segmentation** | [Paper](RanPaste_Paste_Consistency_and_Pseudo_Label_for_Semisupervised_Remote_Sensing_Image_Semantic_Segmentation.pdf) | - | GRS | 2021 | [BibTeX](assets/bibtex/wang2021ranpaste.txt) |

### 1.6 Test-Time Adaptation & Continual Learning

|  | Method | Title | Paper | Code | Venue | Year | BibTeX |
|:--:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| ⭐ | - | **Source-Free Online Domain Adaptive Semantic Segmentation of Satellite Images Under Image Degradation** | [Paper](Source-Free_Online_Domain_Adaptive_Semantic_Segmentation_of_Satellite_Images_Under_Image_Degradation.pdf) | [Code](https://sat-tta.github.io/) | ICASSP | 2024 | [BibTeX](assets/bibtex/niloy2024source.txt) |
| ⭐ | - | **Privacy-Preserving Synthetic Continual Semantic Segmentation for Robotic Surgery** | [Paper](Privacy-Preserving_Synthetic_Continual_Semantic_Segmentation_for_Robotic_Surgery.pdf) | [Code](https://github.com/XuMengyaAmy/Synthetic_CAT_SD) | IEEE | 2024 | [BibTeX](assets/bibtex/xu2024privacy.txt) |
| ⭐ | TTA | **Test-Time Adaptation for Nighttime Color-Thermal Semantic Segmentation** | [Paper](Test-Time_Adaptation_for_Nighttime_Color-Thermal_Semantic_Segmentation.pdf) | [Code](https://vlis2022.github.io/nighttta) | TAI | 2023 | [BibTeX](assets/bibtex/liu2023test.txt) |
| ⭐ | multi-scale TTA | **Top-K Confidence Map Aggregation for Robust Semantic Segmentation Against Unexpected Degradation** | [Paper](Top-K_Confidence_Map_Aggregation_for_Robust_Semantic_Segmentation_Against_Unexpected_Degradation.pdf) | [Code](http://www.ok.sc.e.titech.ac.jp/res/CNNIR/IRDI/) | ICCE | 2023 | [BibTeX](assets/bibtex/moriyasu2023top.txt) |
| ⭐ | HAMLET | **To Adapt or Not to Adapt? Real-Time Adaptation for Semantic Segmentation** | [Paper](Colomer_To_Adapt_or_Not_to_Adapt_Real-Time_Adaptation_for_Semantic_ICCV_2023_paper.pdf) | [Code](https://marcbotet.github.io/hamlet-web/) | ICCV | 2023 | [BibTeX](assets/bibtex/colomer2023adapt.txt) |
|  | PAN | **Enhanced Model Robustness to Input Corruptions by Per-corruption Adaptation of Normalization Statistics** | [Paper](Enhanced_Model_Robustness_to_Input_Corruptions_by_Per-corruption_Adaptation_of_Normalization_Statistics.pdf) | - | IROS | 2024 | [BibTeX](assets/bibtex/camuffo2024enhanced.txt) |
|  | - | **Test-time Training for Matching-based Video Object Segmentation** | [Paper](NeurIPS-2023-test-time-training-for-matching-based-video-object-segmentation-Paper-Conference.pdf) | - | ANIPS | 2023 | [BibTeX](assets/bibtex/bertrand2023test.txt) |
|  | - | **Principles of forgetting in domain-incremental semantic segmentation in adverse weather conditions** | [Paper](Kalb_Principles_of_Forgetting_in_Domain-Incremental_Semantic_Segmentation_in_Adverse_Weather_CVPR_2023_paper.pdf) | - | CV | 2023 | [BibTeX](assets/bibtex/kalb2023principles.txt) |
|  | EndoCSS | **Rethinking exemplars for continual semantic segmentation in endoscopy scenes: Entropy-based mini-batch pseudo-replay** | [Paper](Rethinking%20exemplars%20for%20continual%20semantic%20segmentation%20in%20endoscopy%20scenes%20Entropy-based%20mini-batch%20pseudo-replay.pdf) | - | CBM | 2023 | [BibTeX](assets/bibtex/wang2023rethinking.txt) |
|  | CACE | **Continual Unsupervised Domain Adaptation for Semantic Segmentation using a Class-Specific Transfer** | [Paper](Continual_Unsupervised_Domain_Adaptation_for_Semantic_Segmentation_using_a_Class-Specific_Transfer.pdf) | - | IJCNN | 2022 | [BibTeX](assets/bibtex/marsden2022continual.txt) |
|  | M-ADA | **Learning to Learn Single Domain Generalization** | [Paper](Qiao_Learning_to_Learn_Single_Domain_Generalization_CVPR_2020_paper.pdf) | - | CVPR | 2020 | [BibTeX](assets/bibtex/qiao2020learning.txt) |


## 2. Joint Restoration & Segmentation

### 2.3 Denoising + Segmentation

|  | Method | Title | Paper | Code | Venue | Year | BibTeX |
|:--:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| ⭐ | AATCT-IDS | **AATCT-IDS: A benchmark Abdominal Adipose Tissue CT Image Dataset for image denoising, semantic segmentation, and radiomics evaluation** | [Paper](AATCT-IDS%20A%20benchmark%20Abdominal%20Adipose%20Tissue%20CT%20Image%20Dataset%20for%20image%20denoising,%20semantic%20segmentation,%20and%20radiomics%20evaluation.pdf) | [Code](https://figshare.com/articles/dataset/AATTCT-IDS/23807256) | CBM | 2024 | [BibTeX](assets/bibtex/ma2024aatct.txt) |
| ⭐ | IA-Seg | **Improving Nighttime Driving-Scene Segmentation via Dual Image-Adaptive Learnable Filters** | [Paper](Improving_Nighttime_Driving-Scene_Segmentation_via_Dual_Image-Adaptive_Learnable_Filters.pdf) | [Code](https://github.com/wenyyu/IA-Seg) | TCSVT | 2023 | [BibTeX](assets/bibtex/liu2023improving.txt) |
|  | ICER-Net | **Multi task deep learning phase unwrapping method based on semantic segmentation** | [Paper](Wang_2024_J._Opt._26_115709.pdf) | - | IOP | 2024 | [BibTeX](assets/bibtex/wang2024multi.txt) |
|  | SARDeSeg | **Segmentation-Guided Semantic-Aware Self-Supervised Denoising for SAR Image** | [Paper](Segmentation-Guided_Semantic-Aware_Self-Supervised_Denoising_for_SAR_Image.pdf) | - | GRS | 2023 | [BibTeX](assets/bibtex/yuan2023segmentation.txt) |
|  | DCAIP | **Speckle Reduction via Deep Content-Aware Image Prior for Precise Breast Tumor Segmentation in an Ultrasound Image** | [Paper](Speckle_Reduction_via_Deep_Content-Aware_Image_Prior_for_Precise_Breast_Tumor_Segmentation_in_an_Ultrasound_Image.pdf) | - | UFFC | 2022 | [BibTeX](assets/bibtex/lee2022speckle.txt) |
|  | DDeP | **Denoising Pretraining for Semantic Segmentation** | [Paper](Brempong_Denoising_Pretraining_for_Semantic_Segmentation_CVPRW_2022_paper.pdf) | - | CVPR | 2022 | [BibTeX](assets/bibtex/brempong2022denoising.txt) |
|  | EFNet | **EFNet: Enhancement-Fusion Network for Semantic Segmentation** | [Paper](EFNet%20Enhancement-Fusion%20Network%20for%20Semantic%20Segmentation.pdf) | - | PR | 2021 | [BibTeX](assets/bibtex/wang2021efnet.txt) |
|  | DN-GAN | **DN-GAN: Denoising generative adversarial networks for speckle noise reduction in optical coherence tomography images** | [Paper](DN-GAN%20Denoising%20generative%20adversarial%20networks%20for%20speckle%20noise%20reduction%20in%20optical%20coherence%20tomography%20images.pdf) | - | BSPC | 2020 | [BibTeX](assets/bibtex/chen2020dn.txt) |
|  | - | **Improved denoising autoencoder for maritime image denoising and semantic segmentation of USV** | [Paper](Improved_denoising_autoencoder_for_maritime_image_denoising_and_semantic_segmentation_of_USV.pdf) | - | IEEE | 2020 | [BibTeX](assets/bibtex/qiu2020improved.txt) |
|  | DAPAS | **DAPAS : Denoising Autoencoder to Prevent Adversarial attack in Semantic Segmentation** | [Paper](DAPAS__Denoising_Autoencoder_to_Prevent_Adversarial_attack_in_Semantic_Segmentation.pdf) | - | IJCNN | 2020 | [BibTeX](assets/bibtex/cho2020dapas.txt) |
|  | - | **Effective image restoration for semantic segmentation** | [Paper](Effective%20image%20restoration%20for%20semantic%20segmentation.pdf) | - | Neurocomputing | 2020 | [BibTeX](assets/bibtex/niu2020effective.txt) |
|  | - | **Cooperative Semantic Segmentation and Image Restoration in Adverse Environmental Conditions** | [Paper](Cooperative%20Semantic%20Segmentation%20and%20Image%20Restoration%20in%20Adverse%20Environmental%20Conditions.pdf) | - | arXiv | 2019 | [BibTeX](assets/bibtex/xia2019cooperative.txt) |


## 1. Domain Adaptation & Generalization (DA/DG)

### 1.1 Adversarial Learning Approaches

|  | Method | Title | Paper | Code | Venue | Year | BibTeX |
|:--:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| ⭐ | ICDA | **ICDA: Illumination-Coupled Domain Adaptation Framework for Unsupervised Nighttime Semantic Segmentation** | [Paper](ICDA%20Illumination-Coupled%20Domain%20Adaptation%20Framework%20for%20Unsupervised%20Nighttime%20Semantic%20Segmentation.pdf) | [Code](https://github.com/chenghaoDong666/ICDA) | IJCAI | 2023 | [BibTeX](assets/bibtex/dong2023icda.txt) |
| ⭐ | CMA | **Contrastive model adaptation for cross-condition robustness in semantic segmentation** | [Paper](Bruggemann_Contrastive_Model_Adaptation_for_Cross-Condition_Robustness_in_Semantic_Segmentation_ICCV_2023_paper.pdf) | [Code](https://github.com/brdav/cma) | ICCV | 2023 | [BibTeX](assets/bibtex/bruggemann2023contrastive.txt) |
| ⭐ | Heatnet | **Heatnet: Bridging the day-night domain gap in semantic segmentation with thermal images** | [Paper](HeatNet_Bridging_the_Day-Night_Domain_Gap_in_Semantic_Segmentation_with_Thermal_Images.pdf) | [Code](http://thermal.cs.uni-freiburg.de/) | IROS | 2020 | [BibTeX](assets/bibtex/vertens2020heatnet.txt) |
|  | DBTS | **Dual-branch teacher-student with noise-tolerant learning for domain adaptive nighttime segmentation** | [Paper](Dual-branch%20teacher-student%20with%20noise-tolerant%20learning%20for%20domain%20adaptive%20nighttime%20segmentation.pdf) | - | IVC | 2024 | [BibTeX](assets/bibtex/chen2024dual.txt) |
|  | - | **Weakly supervised semantic segmentation for point cloud based on view-based adversarial training and self-attention fusion** | [Paper](Weakly%20supervised%20semantic%20segmentation%20for%20point%20cloud%20based%20on%20view-based%20adversarial%20training%20and%20self-attention%20fusion.pdf) | - | Computers & Graphics | 2023 | [BibTeX](assets/bibtex/miao2023weakly.txt) |
|  | IEC-Net | **All-weather road drivable area segmentation method based on CycleGAN** | [Paper](All-weather%20road%20drivable%20area%20segmentation%20method%20based%20on%20CycleGAN.pdf) | - | VC | 2023 | [BibTeX](assets/bibtex/jiqing2023all.txt) |
|  | FISS GAN | **FISS GAN: A generative adversarial network for foggy image semantic segmentation** | [Paper](FISS_GAN_A_Generative_Adversarial_Network_for_Foggy_Image_Semantic_Segmentation.pdf) | - | JAS | 2021 | [BibTeX](assets/bibtex/liu2021fiss.txt) |
|  | W-GAN | **Semantic Segmentation With Unsupervised Domain Adaptation Under Varying Weather Conditions for Autonomous Vehicles** | [Paper](Semantic_Segmentation_With_Unsupervised_Domain_Adaptation_Under_Varying_Weather_Conditions_for_Autonomous_Vehicles.pdf) | - | RAL | 2020 | [BibTeX](assets/bibtex/erkent2020semantic.txt) |
|  | RSSN | **Nighttime Road Scene Parsing by Unsupervised Domain Adaptation** | [Paper](Nighttime_Road_Scene_Parsing_by_Unsupervised_Domain_Adaptation.pdf) | - | TITS | 2020 | [BibTeX](assets/bibtex/song2020nighttime.txt) |

### 1.5 Knowledge Distillation

|  | Method | Title | Paper | Code | Venue | Year | BibTeX |
|:--:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| ⭐ | - | **Lightweight deep learning methods for panoramic dental X-ray image segmentation** | [Paper](Lightweight%20deep%20learning%20methods%20for%20panoramic%20dental%20X-ray%20image%20segmentation.pdf) | [Code](https://github.com/IvisionLab/dental-image.) | NCA | 2023 | [BibTeX](assets/bibtex/lin2023lightweight.txt) |
| ⭐ | - | **Unsupervised foggy scene understanding via self spatial-temporal label diffusion** | [Paper](Unsupervised_Foggy_Scene_Understanding_via_Self_Spatial-Temporal_Label_Diffusion.pdf) | [Code](http://people.ee.ethz.ch/∼ csakarid/SFSU_synthetic/) | TIP | 2022 | [BibTeX](assets/bibtex/liao2022unsupervised.txt) |
| ⭐ | Heatnet | **Heatnet: Bridging the day-night domain gap in semantic segmentation with thermal images** | [Paper](HeatNet_Bridging_the_Day-Night_Domain_Gap_in_Semantic_Segmentation_with_Thermal_Images.pdf) | [Code](http://thermal.cs.uni-freiburg.de/) | IROS | 2020 | [BibTeX](assets/bibtex/vertens2020heatnet.txt) |
|  | MTKD | **Weather-degraded image semantic segmentation with multi-task knowledge distillation** | [Paper](Weather-degraded%20image%20semantic%20segmentation%20with%20multi-task%20knowledge%20distillation.pdf) | - | IV | 2022 | [BibTeX](assets/bibtex/li2022weather.txt) |
|  | multi-teacher KD | **Robust Semantic Segmentation With Multi-Teacher Knowledge Distillation** | [Paper](Robust_Semantic_Segmentation_With_Multi-Teacher_Knowledge_Distillation.pdf) | - | IEEE | 2021 | [BibTeX](assets/bibtex/amirkhani2021robust.txt) |
|  | - | **Efficient Uncertainty Estimation in Semantic Segmentation via Distillation** | [Paper](Holder_Efficient_Uncertainty_Estimation_in_Semantic_Segmentation_via_Distillation_ICCVW_2021_paper.pdf) | - | ICCV | 2021 | [BibTeX](assets/bibtex/holder2021efficient.txt) |


## 3. Multi-Modal Fusion

### 3.1 RGB + Thermal Fusion

|  | Method | Title | Paper | Code | Venue | Year | BibTeX |
|:--:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| ⭐ | CMNeXt, DeLiVER | **Delivering Arbitrary-Modal Semantic Segmentation** | [Paper](Zhang_Delivering_Arbitrary-Modal_Semantic_Segmentation_CVPR_2023_paper.pdf) | [Code](https://jamycheung.github.io/DELIVER.html) | CVPR | 2023 | [BibTeX](assets/bibtex/zhang2023delivering.txt) |
| ⭐ | MCNet | **MCNet: Multi-level Correction Network for thermal image semantic segmentation of nighttime driving scene** | [Paper](MCNet%20Multi-level%20Correction%20Network%20for%20thermal%20image%20semantic%20segmentation%20of%20nighttime%20driving%20scene.pdf) | [Code](https://github.com/haitaobiyao/MCNet) | INFRARED PHYS TECHN‌‌ | 2021 | [BibTeX](assets/bibtex/xiong2021mcnet.txt) |
|  | CMEFNet | **Illumination Robust Semantic Segmentation Based on Cross-Dimensional Multispectral Edge Fusion in Dynamic Traffic Scenes** | [Paper](Illumination_Robust_Semantic_Segmentation_Based_on_Cross-Dimensional_Multispectral_Edge_Fusion_in_Dynamic_Traffic_Scenes.pdf) | - | IEEE | 2024 | [BibTeX](assets/bibtex/ni2024illumination.txt) |
|  | - | **Weakly supervised semantic segmentation for point cloud based on view-based adversarial training and self-attention fusion** | [Paper](Weakly%20supervised%20semantic%20segmentation%20for%20point%20cloud%20based%20on%20view-based%20adversarial%20training%20and%20self-attention%20fusion.pdf) | - | Computers & Graphics | 2023 | [BibTeX](assets/bibtex/miao2023weakly.txt) |
|  | - | **Test-time Training for Matching-based Video Object Segmentation** | [Paper](NeurIPS-2023-test-time-training-for-matching-based-video-object-segmentation-Paper-Conference.pdf) | - | ANIPS | 2023 | [BibTeX](assets/bibtex/bertrand2023test.txt) |
|  | FuseSeg | **FuseSeg: Semantic Segmentation of Urban Scenes Based on RGB and Thermal Data Fusion** | [Paper](FuseSeg_Semantic_Segmentation_of_Urban_Scenes_Based_on_RGB_and_Thermal_Data_Fusion.pdf) | - | TASE | 2020 | [BibTeX](assets/bibtex/sun2020fuseseg.txt) |
|  | RTFNet | **RTFNet: RGB-Thermal Fusion Network for Semantic Segmentation of Urban Scenes** | [Paper](RTFNet_RGB-Thermal_Fusion_Network_for_Semantic_Segmentation_of_Urban_Scenes.pdf) | - | RAL | 2019 | [BibTeX](assets/bibtex/sun2019rtfnet.txt) |


## 1. Domain Adaptation & Generalization (DA/DG)

### 1.7 Other DA/DG Strategies

|  | Method | Title | Paper | Code | Venue | Year | BibTeX |
|:--:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| ⭐ | - | **Self Pseudo Entropy Knowledge Distillation for Semi-Supervised Semantic Segmentation** | [Paper](Self_Pseudo_Entropy_Knowledge_Distillation_for_Semi-Supervised_Semantic_Segmentation.pdf) | [Code](https://github.com/xiaoqiang-lu/SPEED) | CVPR | 2022 | [BibTeX](assets/bibtex/bar2022performance.txt) |
| ⭐ | SDBF | **Curriculum Model Adaptation with Synthetic and Real Data for Semantic Foggy Scene Understanding** | [Paper](Curriculum%20Model%20Adaptation%20with%20Synthetic%20and%20Real%20Data%20for%20Semantic%20Foggy%20Scene%20Understanding.pdf) | [Code](https://people.ee.ethz.ch/~csakarid/Model_adaptation_SFSU_dense/) | IJCV | 2020 | [BibTeX](assets/bibtex/dai2020curriculum.txt) |
|  | MGML | **Complementary Masked-Guided Meta-Learning for Domain Adaptive Nighttime Segmentation** | [Paper](Complementary_Masked-Guided_Meta-Learning_for_Domain_Adaptive_Nighttime_Segmentation.pdf) | - | IEEE | 2024 | [BibTeX](assets/bibtex/chen2024complementary.txt) |
|  | - | **Improving semantic segmentation under hazy weather for autonomous vehicles using explainable artificial intelligence and adaptive dehazing approach** | [Paper](Improving_Semantic_Segmentation_Under_Hazy_Weather_for_Autonomous_Vehicles_Using_Explainable_Artificial_Intelligence_and_Adaptive_Dehazing_Approach.pdf) | - | IEEE | 2023 | [BibTeX](assets/bibtex/saravanarajan2023improving.txt) |


## 2. Joint Restoration & Segmentation

### 2.1 Dehazing/Defogging+ Segmentation

|  | Method | Title | Paper | Code | Venue | Year | BibTeX |
|:--:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| ⭐ | - | **Towards Robust Rain Removal Against Adversarial Attacks: A Comprehensive Benchmark Analysis and Beyond** | [Paper](Yu_Towards_Robust_Rain_Removal_Against_Adversarial_Attacks_A_Comprehensive_Benchmark_CVPR_2022_paper.pdf) | [Code](https://github.com/yuyisd/Robust_Rain_Removal) | CVPR | 2022 | [BibTeX](assets/bibtex/yu2022towards.txt) |
| ⭐ | SNE-Seg | **SS-SFDA: Self-Supervised Source-Free Domain Adaptation for Road Segmentation in Hazardous Environments** | [Paper](Kothandaraman_SS-SFDA_Self-Supervised_Source-Free_Domain_Adaptation_for_Road_Segmentation_in_Hazardous_ICCVW_2021_paper.pdf) | [Code](https://gamma.umd.edu/weatherSAfE/) | ICCV | 2021 | [BibTeX](assets/bibtex/kothandaraman2021ss.txt) |
|  | - | **Budget-Aware Road Semantic Segmentation in Unseen Foggy Scenes** | [Paper](Budget-Aware_Road_Semantic_Segmentation_in_Unseen_Foggy_Scenes.pdf) | - | RIVF | 2023 | [BibTeX](assets/bibtex/to2023budget.txt) |
|  | - | **Cooperative Semantic Segmentation and Image Restoration in Adverse Environmental Conditions** | [Paper](Cooperative%20Semantic%20Segmentation%20and%20Image%20Restoration%20in%20Adverse%20Environmental%20Conditions.pdf) | - | arXiv | 2019 | [BibTeX](assets/bibtex/xia2019cooperative.txt) |
|  | - | **A Convolutional Network for Joint Deraining and Dehazing from A Single Image for Autonomous Driving in Rain** | [Paper](A_Convolutional_Network_for_Joint_Deraining_and_Dehazing_from_A_Single_Image_for_Autonomous_Driving_in_Rain.pdf) | - | IROS | 2019 | [BibTeX](assets/bibtex/sun2019convolutional.txt) |

### 2.7 JPEG Decoding + Segmentation

|  | Method | Title | Paper | Code | Venue | Year | BibTeX |
|:--:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| ⭐ | DSSLIC | **DSSLIC: Deep Semantic Segmentation-based Layered Image Compression** | [Paper](DSSLIC_Deep_Semantic_Segmentation-based_Layered_Image_Compression.pdf) | [Code](https://github.com/makbari7/DSSLIC) | ICASSP | 2019 | [BibTeX](assets/bibtex/akbari2019dsslic.txt) |
|  | DCT-CompSegNet | **DCT-CompSegNet: fast layout segmentation in DCT compressed JPEG document images using deep feature learning** | [Paper](DCT-CompSegNet%20fast%20layout%20segmentation%20in%20DCT%20compressed%20JPEG%20document%20images%20using%20deep%20feature%20learning.pdf) | - | MTA | 2024 | [BibTeX](assets/bibtex/rajesh2024dct.txt) |
|  | - | **Semantic segmentation in learned compressed domain** | [Paper](Semantic_Segmentation_In_Learned_Compressed_Domain.pdf) | - | PCS | 2022 | [BibTeX](assets/bibtex/liu2022semantic.txt) |
|  | ERA | **Reverse Error Modeling for Improved Semantic Segmentation** | [Paper](Reverse_Error_Modeling_for_Improved_Semantic_Segmentation.pdf) | - | ICIP | 2022 | [BibTeX](assets/bibtex/kuhn2022reverse.txt) |
|  | CCAFFMNet | **CCAFFMNet: Dual-spectral semantic segmentation network with channel-coordinate attention feature fusion module** | [Paper](CCAFFMNet%20Dual-spectral%20semantic%20segmentation%20network%20with%20channel-coordinate%20attention%20feature%20fusion%20module.pdf) | - | Neurocomputing | 2022 | [BibTeX](assets/bibtex/yi2022ccaffmnet.txt) |
|  | - | **Deep Learning Based Image Segmentation Directly in the JPEG Compressed Domain** | [Paper](Deep_Learning_Based_Image_Segmentation_Directly_in_the_JPEG_Compressed_Domain.pdf) | - | UPCON | 2021 | [BibTeX](assets/bibtex/singh2021deep.txt) |

### 2.4 Deblurring + Segmentation

|  | Method | Title | Paper | Code | Venue | Year | BibTeX |
|:--:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| ⭐ | - | **From Motion Blur to Motion Flow: a Deep Learning Solution for Removing Heterogeneous Motion Blur** | [Paper](Gong_From_Motion_Blur_CVPR_2017_paper.pdf) | [Code](https://donggong1.github.io/blur2mflow) | CVPR | 2017 | [BibTeX](assets/bibtex/gong2017motion.txt) |
|  | - | **Effective image restoration for semantic segmentation** | [Paper](Effective%20image%20restoration%20for%20semantic%20segmentation.pdf) | - | Neurocomputing | 2020 | [BibTeX](assets/bibtex/niu2020effective.txt) |

### 2.6 Low-Light Enhancement + Segmentation

|  | Method | Title | Paper | Code | Venue | Year | BibTeX |
|:--:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|  | - | **Semantic segmentation of JPEG blocks using a deep CNN for non-aligned JPEG forgery detection and localization** | [Paper](Semantic%20segmentation%20of%20JPEG%20blocks%20using%20a%20deep%20CNN%20for%20non-aligned%20JPEG%20forgery%20detection%20and%20localization.pdf) | - | MTA | 2020 | [BibTeX](assets/bibtex/alipour2020semantic.txt) |

---

## 📚 Citation

If you find this survey helpful, please cite:

```bibtex
@article{chen2025degraded,
  title={A Survey on Degraded Image Segmentation},
  author={Chen, Linwei and Fu, Ying and Shangguan, Jingyu and Xu, Jinglin and Peng, Yuxin},
  journal={Chinese Journal of Electronics},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request to add new papers or fix any issues.

## 📧 Contact

- **Linwei Chen** - Beijing Institute of Technology
- **Ying Fu** (Corresponding Author) - fuying@bit.edu.cn

---

<div align="center">
Made with ❤️ for the Computer Vision Community
</div>
