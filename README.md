# CAMCNs-Causal-Discovery
A Time Series Causal Discovery Model with attention and multilag selection

This project implements a Convolutional and Multihead Attention Neural Network with non-linear correlation (CAMCNs)  for causal inference in time series data. The primary goal is to identify causal relationships and their corresponding time delays between multiple time series.

## File Descriptions

### `run.py`
- The main script to execute the causal discovery and evaluation process.
- Contains the `CDCNN` class, which encapsulates the entire process including training, evaluation, and visualization.

### `data.py`
- Handles data processing tasks.
- Defines the `DataProcess` class for loading and preprocessing data.
- Methods:
  - `df_to_dict`: Converts a DataFrame to a dictionary.
  - `prepare_data`: Normalizes data and prepares it for model input.

### `main.py`
- Orchestrates the workflow by utilizing the `DataProcess` class for data handling and `CDCNN` class for model execution.
- Sets up parameters and initiates the causal discovery process.
- Example usage is provided to demonstrate how to run the model and evaluate its performance.

### `model_code.py`
- Defines the neural network architecture and training procedures.
- Contains various PyTorch models such as `FirstBlock`, `TemporalBlock`, `LastBlock`, and `DepthwiseNet`.
- `ADDSTCN` class implements the main Temporal Convolutional Network (TCN) with additional components for causal discovery.
- `TCFMODEL` class manages training and validation routines.

## Setup and Installation

### Requirements

- Python 3.x
- Required libraries: `torch`, `pandas`, `numpy`, `scikit-learn`, `networkx`, `matplotlib`, `argparse`, `scipy`, `statsmodels`

### Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

1. Ensure your data files are in the appropriate format:
   - `.xlsx` for primary data.
   - `.csv` for evaluation data.

2. Modify the file paths in `main.py` to point to your data files:
   ```python
   data_handler = DataProcess('<path_to_xlsx>', '<path_to_csv>')
   ```

### Running the Model

1. Execute the `main.py` script:
   ```bash
   python main.py
   ```

2. The script will load and preprocess the data, configure the model parameters, and start the causal discovery process.

### Visualization

- The discovered causal relationships and their delays are visualized using NetworkX and Matplotlib. The results are displayed as a temporal causal graph.

## Configuration

- Model parameters can be adjusted in `main.py` to fine-tune the performance:
  ```python
  dilatation_c = 2
  valp = 0.2
  kernel_size = 4  
  levels = 1
  nrepochs = 500
  valepoc = 200
  learningrate = 0.001
  optimizername = "Adam"
  dilation_c = 1
  loginterval = 500
  seed=1234
  cuda=True
  significance=0.99
  plot=True
  lags=5
  panel=False
  ```

## Evaluation

- The model's performance is evaluated based on precision, recall, F1-score, and accuracy.
- Evaluation results are printed and can be compared against ground truth data provided in the evaluation CSV file.

## Contribution

- Contributions are welcome! Please open an issue or submit a pull request.

## License

- This project is licensed under the GPL-3 License.
