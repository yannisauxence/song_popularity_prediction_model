# Song Popularity Prediction

This project aims to predict the popularity of songs using features extracted from a dataset. By leveraging machine learning techniques, we analyze patterns and build a predictive model.

## Project Overview
The goal of this project is to create a predictive model to estimate the popularity of songs based on their attributes. The dataset includes various song features, such as tempo, energy, and danceability, which are used as inputs to train and evaluate the model.
You can find the presenation in `presentation/`
## Setup Instructions

### Prerequisites
Ensure you have Python 3.7 or later installed on your system. It`s recommended to use a virtual environment to manage dependencies.

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/yannisauxence/song_popularity_prediction_model.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Look for the dataset in the root directory of the project. Ensure the file is named `data.csv`. Or download it [here](https://www.kaggle.com/datasets/thebumpkin/10400-classic-hits-10-genres-1923-to-2023)

## How to Run
1. Open the notebook `demo.ipynb` in Jupyter Notebook or JupyterLab located in the `demo/` folder

2. Execute the cells sequentially to:
    - Load and preprocess the sample dataset.
    - Make predictions on sample data.
    - Save results in the `results/` folder

Alternatively, run (`demo.py`) it directly:
```bash
python demo/demo.py
```
3. In the `src/` folder you will find thr codebase for the model training in either `main.py` or `main.ipynb` (added in cae `main.py` does not work)

## Expected Output
1. Running the demo script will output a visualization of the first row of the testing dataset, to make sure the data was loaded properly.

2. Running `main.py` or `main.ipynb` will train the model and output the model`s training loss, testing loss, and number of correct predictions

## Pre-trained Model Link
You can find the pre_trained model in the "checkpoint/" directory, or you can download it [here](https://huggingface.co/yannisauxence/song_pop/tree/main) and place it in the `checkpoint/` directory.

## Acknowledgments
- Dataset sourced from [Kaggle](https://www.kaggle.com/datasets/thebumpkin/10400-classic-hits-10-genres-1923-to-2023).
- Libraries and frameworks: NumPy, pandas, PyTorch, SciKit-Learn, Matplotlib, and Seaborn.
- Special thanks to my instructors Jimin Kim and Yang Zheng for their guidance and feedback, and the machine learning community for open-source resources.


