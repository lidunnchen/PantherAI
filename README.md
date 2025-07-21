# PantherAI: an autonomous behavioural monitoring tool for assessing activity budget and space use in a zoo-housed tiger
Li-Dunn Chen, Stephen Dodds, Molly McGuire, Maria Frankea, Gabriela Mastromonaco


## Description
This repository contains code for the PantherAI behavioural monitoring framework, a computer vision method that can be used for real-time monitoring of CCTV livestreams for generating metrics such as activity budget and space use heatmaps.

Please refer to the ["Documentation"](https://alexhang212.github.io/YOLO_Behaviour_Repo/) for full installation and implementation guidelines, and the [paper](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14502) for detailed description of the method!

![banner](./Images/GraphicalAbstract.png)


## Abstract

> Machine learning (ML)-aided technologies can be applied to many of the existing wildlife science tools (e.g., camera traps) used to support conservation initiatives both in situ and ex situ. The automated nature of ML methods reduces manual labour, extends monitoring efforts past regular daylight/working hours, and improves the overall diagnostic capacity of tools routinely applied by wildlife biologists and animal care staff at zoological institutions. Though the conservation aims and expectations may differ among zoos and aquariums, simple monitoring tools that impose less demand on animal care staff should serve as an important aid for advancing management strategies for threatened species. We applied computer vision-based predictive models built on CCTV footage from zoo-housed Panthera tigris to develop an automated behavioural monitoring tool (“PantherAI”) capable of rapidly assessing activity budget and space use across variable lighting and weather conditions. We applied YOLOv8 as the model backbone to detect and classify several tiger behaviours (e.g., stereotypical pacing, resting, enrichment interaction, feeding); the trained models were then applied with scripts to autonomously generate customized activity budgets and space use heatmaps from 24-hour video samples. PantherAI yielded a mean average precision >75% on test data, where it detected and classified tiger behaviours with varying levels of accuracy (stereotypical pacing: 100%, resting: 84%, locomotion: 81%, feeding: 43%, object manipulation = 40%). Activity budgets varied (p<0.05) across habitats and by time of day for several behaviours. PantherAI provided reliable estimates of behaviour and space usage, two important ecological metrics commonly used to establish baseline activity budgets and assess indicators of animal welfare. Overall, ML-coupled technologies can facilitate daily data collection and monitoring procedures, both of which are integral for objectively measuring behavioural outcomes as newly implemented husbandry practices (e.g., alterations to diet, environment, social group, enrichment) are enacted in zoological and other ex situ conservation settings. 


## Quick Start
We provide a whole pipeline from data annotation to model training to inference in the [Documentation](INSERT LINK TO REPO). Here, we will run a quick demo inference visualization. Make sure you download the [sample dataset](INSERT LINK FOR SMALL DATASET INSTALLATION), and place it under the `Data/` directory. In the future, video walkthroughs will be provided.

### Installation
There are a series of required packages to run the pipeline. We recommend creating a [conda environment](https://www.anaconda.com/). 

You can create a new environment and install required packages by running:
```
conda create -n YOLO python=3.8
conda activate YOLO

pip install -r requirements.txt
```

### Run Inference on Sample Data
After installation and downloading the sample dataset, run this in the terminal, making sure that the current working directory is in `PantherAI_Repo`. You can change your working directory by using the "cd" command:  `cd /path/to/PantherAI_Repo`

Please refer to the manuscript cited below for details regarding the YOLO file structure for deploying the PantherAI pipeline. Note several files and associated scripts are needed to 1) preprocess data, 2) prepare data for analysis, 3) train models, and 4) deploy models for generating activity budget plots and space use heatmaps. 
![PantherAI Scripts](./Images/Figure3.png)
```
python Code/3_VisualizeResults.py --Video "./Data/JaySampleData/Jay_Sample.mp4" --Weight  "./Data/Weights/JayBest.pt" --Start 0 --Frames -1

```
[![Watch the video](https://github.com/lidunnchen/PantherAI/blob/main/Images/Video1_Still.png)](https://github.com/lidunnchen/PantherAI/blob/main/Images/SupplementalVideo1_c28_loco_obman.mp4)


## Citation
```
```



## Contact
If you have any questions/ suggestions with the pipeline, or any additional instructions/ guidelines you would like to see in the documentation, feel free to leave a git issue or contact me via email:

lchen[at]torontozoo.ca
