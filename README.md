# Repository for Land Cover Recognition Model Training Using Satellite Imagery

Welcome to the dedicated repository for advancing land cover recognition through the application of state-of-the-art models on satellite imagery. This repository serves as a comprehensive resource for researchers and practitioners in the field, providing access to research code, detailed setup instructions, and guidelines for conducting experiments with satellite image timeseries data.

## Urban Adaptation Project (2025)

This repository has been extended to include an **Urban Computing project** that adapts the TSViT (Temporo-Spatial Vision Transformer) architecture from agricultural applications to urban satellite image time series analysis.

### Project Overview

**Title**: Adapting Vision Transformers for Urban Satellite Image Time Series

**Research Question**: Does the temporal-first attention factorization of TSViT, originally designed for agricultural land cover classification, remain effective when applied to urban environments where spatial structure is highly informative?

### Key Contributions

- **Urban Dataset Integration**: Adapted TSViT to work with the SpaceNet7 multi-temporal urban development dataset for building detection
- **Architectural Comparison**: Implemented and compared temporal-first (TSViT) vs. spatial-first (TSViT-ST) attention factorization strategies
- **Comprehensive Evaluation**: Conducted systematic experiments with 3 training runs per model to assess stability and performance
- **Baseline Comparison**: Benchmarked against 3D convolutional architectures (UNet3D)

### Main Findings

- **TSViT (Temporal-First)**: Maintains stable performance on urban data (IoU = 0.646 ± 0.031)
- **TSViT-ST (Spatial-First)**: Exhibits severe training instability with 2/3 runs failing to converge (IoU < 0.10)
- **UNet3D**: Achieves best performance (IoU = 0.672 ± 0.014), demonstrating the continued relevance of 3D convolutions for urban spatio-temporal modeling

### Urban-Specific Components

- **Data Pipeline**: `data/SpaceNet7/` - Custom dataloader, preprocessing, and transformations for SpaceNet7 dataset
- **Model Variants**: `models/TSViT/TSViTdense_ST.py` - Spatial-first attention variant implementation
- **Configuration Files**: `configs/SpaceNet7/` - Experiment configurations for all model variants
- **Evaluation Scripts**: `scripts/evaluate_on_test.py`, `scripts/analyze_results.py` - Comprehensive evaluation and statistical analysis
- **Results**: `results/analysis/` - Performance metrics, visualizations, and statistical tests

### Quick Start for Urban Experiments

1. **Setup Environment**: Follow the environment setup instructions below
2. **Prepare SpaceNet7 Data**: Place the SpaceNet7 dataset in `data/SpaceNet7/train/`
3. **Run Training**: Use configurations in `configs/SpaceNet7/` for different model variants
4. **Evaluate**: Run `scripts/evaluate_on_test.py` to generate test metrics
5. **Analyze**: Use `scripts/analyze_results.py` to generate comparison plots and statistical tests

---

## Featured Research Publications

This repository highlights contributions to the field through the following research publications:

- [ViTs for SITS: Vision Transformers for Satellite Image Time Series](https://openaccess.thecvf.com/content/CVPR2023/html/Tarasiou_ViTs_for_SITS_Vision_Transformers_for_Satellite_Image_Time_Series_CVPR_2023_paper.html) - Featured at CVPR 2023, this paper explores the application of Vision Transformers to Satellite Image Time Series analysis. For further details, please consult the [README_TSVIT.md](https://github.com/michaeltrs/DeepSatModels/blob/main/README_TSVIT.md) document.
- [Context-self contrastive pretraining for crop type semantic segmentation](https://ieeexplore.ieee.org/abstract/document/9854891) - 
Published in IEEE Transactions on Geoscience and Remote Sensing, this work introduces a novel supervised pretraining method for semantic segmentation 
of crop types exhibiti performance gains along object boundaries. Additional information is available in the [README_CSCL.md](https://github.com/michaeltrs/DeepSatModels/blob/main/README_CSCL.md) document.

## Environment Setup

### Installation of Miniconda
For the initial setup, please follow the instructions for downloading and installing Miniconda available at the [official Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

### Environment Configuration
1. **Creating the Environment**: Navigate to the code directory in your terminal and create the environment using the provided `.yml` file by executing:

        conda env create -f deepsatmodels_env.yml

2. **Activating the Environment**: Activate the newly created environment with:

        source activate deepsatmodels

3. **PyTorch Installation**: Install the required version of PyTorch along with torchvision and torchaudio by running:

        conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch-nightly

## Alternative Environment Setup (if you dont have Docker or Conda)
1. **Creating the Environment**: Navigate to the code directory in your terminal and create the environment by installing and using [portableenv](https://pypi.org/project/portableenv/)
        
        pip install portableenv

        python -m portableenv .venv
        

2. **Activating the Environment**: Activate the newly created environment with:
        
        .venv\Scripts\activate

3. **Dependencies Installation**: Install the required Python modules

        pip install -r requirements.txt

## Experiment Setup

- **Configuration**: Specify the base directory and paths for training and evaluation datasets within the `data/datasets.yaml` file.
- **Experiment Configuration**: Use a distinct `.yaml` file for each experiment, located in the `configs` folder. These configuration files encapsulate default parameters aligned with those used in the featured research. Modify these `.yaml` files as necessary to accommodate custom datasets.
- **Guidance on Experiments**: For detailed instructions on setting up and conducting experiments, refer to the specific README.MD files associated with each paper or dataset.

## License and Copyright

This project is made available under the Apache License 2.0. Please see the [LICENSE](https://github.com/michaeltrs/DeepSatModels/blob/main/LICENSE.txt) file for detailed licensing information.

Copyright © 2023 by Michail Tarasiou
