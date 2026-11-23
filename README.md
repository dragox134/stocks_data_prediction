# Stock Price Prediction Based on Historical Data
 
A graduation project (Maturita) focused on predicting stock prices using machine learning models.
 
## Project Description
 
The project is a web application built in Python that allows training time series
models on historical stock data.
 
Main goals of the project:
- design and implement a pipeline for loading, preparing and training on data,
- compare real and predicted stock price values,
- visualize the training progress and the quality of the model,
- perform short-term prediction of future values.
Currently implemented models:
- LSTM
- Transformer (option `trs`)
### Technologies Used
 
- Python 3.11
- PyTorch
- Dash
- Plotly
- NumPy
- Pandas
- scikit-learn
- TensorBoard
- yfinance
## Project Structure
 
- `app.py` — web interface, training configuration, display of metrics and charts
- `training_4_0.py` — main training pipeline
- `helper_functions/` — data loading, model definitions, prediction, saving outputs, TensorBoard setup
- `assets/style.css` — styling of the Dash application
- `archived` — a few of the main saved runs
- `logs_pytorch` — all saved runs
- `models` — all saved best models
> **Note:** The repository also contains older scripts for working with local
> SQLite databases and the Alpha Vantage API (`database_scripts/`, `dbs/`, `api/`).
> These components were created as a prototype in the early phase of development and are not part of the
> main functionality of the application — they serve only as a demonstration of the project's initial direction.
> The current version of the application loads data exclusively through the `yfinance` library.
> These files can now be found under the `old/` folder.
 
## Installation and Running
 
### Requirements
 
- Python 3.11 or newer
- pip
### Clone the Repository and Create a Virtual Environment
 
```bash
git clone https://github.com/SPSE-Zoska-IV-C/Gurtner_Predikcia_cien_akcii_na_zaklade_historickych_dat.git
cd Gurtner_Predikcia_cien_akcii_na_zaklade_historickych_dat
python -m venv stock_prediction
```
 
Windows PowerShell:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\stock_prediction\Scripts\Activate.ps1
```
 
Linux/macOS:
```bash
source stock_prediction/bin/activate
```
 
### Install Dependencies
 
```bash
pip install -r requirements.txt
```
 
### Run the Application
 
```bash
python app.py
```
 
After starting, the application is available at `http://127.0.0.1:8050`.
 
## Basic Usage
 
1. In the application, select a model and the training parameters.
2. Start training with the `Train Model` button.
3. Watch the outputs:
   - `Loss/Train`
   - `Loss/Val`
   - `Train/Close`
   - `Test/Close` (including the future prediction)
## Credits
 
- **Project author:** Leo Gürtner
- **Supervisor:** Ing. Oliver Halaš
- **Data source:** Yahoo Finance (via `yfinance`)
## Contributing
 
The project is primarily intended as a school project. Suggestions for improvement are welcome.
 
Recommended process:
1. Fork the repository
2. Create your own branch
3. Open a Pull Request with a description of the changes
## License
 
This project is licensed under the MIT license.
See the `LICENSE` file for details.
 