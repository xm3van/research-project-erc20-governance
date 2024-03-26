# Wallet Projection Experiment Documentation

This repository contains documentation for a scientific experiment conducted to analyze token holdings between wallet pairs using a wallet projection method. The experiment aimed to investigate the intersection of tokens held by different wallets and compute hypergeometric p-values.

## Overview

The Wallet Projection Experiment aims to provide insights into the distribution of tokens across different wallets in a given Ethereum ecosystem. The experiment involved the following steps:

1. Data Loading: Token balance data from CSV files was loaded for analysis.
2. Data Processing: The loaded data was processed to filter out irrelevant information and calculate relevant metrics.
3. Hypergeometric Analysis: The hypergeometric analysis was performed to compute p-values for pairs of wallets.
4. Batch Processing: The combinations of wallet pairs were processed in parallel using multiple threads to improve efficiency.
5. Data Storage: The analysis results were stored in HDF5 format for efficient storage and retrieval.

## Usage

To reproduce the experiment:

1. Clone the repository:

   ```
   git clone https://github.com/your-username/wallet-projection-experiment.git
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Configure the necessary input files:

   - `snapshot_selection.csv`: Specify the snapshots to be analyzed.
   - `df_final_token_selection.csv`: Define the selected tokens for analysis.

4. Run the main script:

   ```
   python main.py
   ```

   This will execute the wallet projection experiment, loading the data, performing the analysis, and storing the results in HDF5 format.

## Project Structure

The project is organized as follows:

- `data_loading/`: Contains modules for loading token balance data.
- `data_processing/`: Contains modules for processing the loaded data.
- `data_storage/`: Contains modules for storing the analysis results.
- `main.py`: The main script that orchestrates the workflow.
- `README.md`: This file providing information about the experiment.

## Results

The experiment results can be found in the `output/` directory. The analysis results for each snapshot are stored in separate HDF5 files named `output_snapshot_<snapshot>.h5`. Additionally, the percentage supply coverage for each snapshot is available in the `pct_supply_coverage.csv` file.

## Contributing

Contributions to this experiment documentation are welcome! If you would like to contribute, please fork the repository and submit a pull request.

## License

This experiment documentation is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Please customize the README file as needed to reflect the specific details of your scientific experiment, such as installation instructions, file paths, and any additional information related to the experiment's objective or hypothesis.