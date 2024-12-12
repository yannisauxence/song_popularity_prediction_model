# Song Popularity Prediction

This project aims to predict the popularity of songs using features extracted from a dataset. By leveraging machine learning techniques, we analyze patterns and build a predictive model.

## Project Overview
The goal of this project is to create a predictive model to estimate the popularity of songs based on their attributes. The dataset includes various song features, such as tempo, energy, and danceability, which are used as inputs to train and evaluate the model.

## Setup Instructions

### Prerequisites
Ensure you have Python 3.7 or later installed on your system. It's recommended to use a virtual environment to manage dependencies.

### Steps
1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset and place it in the root directory of the project. Ensure the file is named `data.csv`.

## How to Run
1. Open the notebook `Project_YannisBoadji.ipynb` in Jupyter Notebook or JupyterLab.

2. Execute the cells sequentially to:
    - Load and preprocess the dataset.
    - Train the model.
    - Make predictions on sample data.

Alternatively, if a script (`demo.py`) is available, run it directly:
```bash
python demo.py
```

## Expected Output
After running the notebook or demo script, the output will include:
- Visualizations of the dataset.
- Model training logs.
- Predictions for song popularity, typically as numerical scores or classifications.

## Pre-trained Model Link
Download the pre-trained model [here](#) and place it in the `models/` directory.

## Acknowledgments
- Dataset sourced from [Spotify Dataset](#).
- Libraries and frameworks: NumPy, pandas, PyTorch, Matplotlib, and Seaborn.
- Special thanks to contributors and the machine learning community for open-source resources.
