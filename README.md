# Emergency Vehicle Sound Classification

This repo contains the code and final presentation for an audio machine learning project that classifies emergency vehicle sirens using a Convolutional Neural Network (CNN). The model was built and fine-tuned using fastai and PyTorch, achieving 99% classification accuracy.

Here are the different components:

- [CNN Notebook](./502_WorkingProgram_CNN.ipynb): Full pipeline including audio preprocessing, feature extraction, model architecture, training, fine-tuning, and evaluation
- [Final Presentation]("./502 Final Project Presentation.pdf"): Project overview, methodology, and results

## Overview

Emergency vehicle detection has real-world applications in traffic management, autonomous vehicles, and accessibility tools. This project builds an audio classification pipeline to distinguish emergency vehicle sirens (ambulance, fire truck, police) from other sounds using a CNN trained on spectrogram representations of audio clips.

- Achieved **99% classification accuracy** through fine-tuning and validation
- Built an end-to-end audio ML pipeline using **fastai** and **PyTorch**
- Converted raw audio into mel spectrograms for image-based CNN classification
- Applied transfer learning and fine-tuning to optimize model performance

## Methods

- **Audio Preprocessing:** Raw audio clips converted to mel spectrograms for visual pattern recognition
- **Model:** Convolutional Neural Network (CNN) built with fastai/PyTorch
- **Training:** Transfer learning with iterative fine-tuning
- **Evaluation:** Accuracy, confusion matrix, and validation loss tracking

## Requirements

- Python 3.x
- Jupyter Notebook

## Dependencies

- torch
- fastai
- torchaudio
- numpy
- matplotlib
- librosa

## Authors

Sophia Huang · Binghamton University MS Data Analytics · Fall 2025
