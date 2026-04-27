# Stock Price Prediction Dashboard

Maturitny projekt zamerany na predikciu cien akcii pomocou modelov strojoveho ucenia.

## 1. Nazov projektu

Stock Price Prediction Dashboard

## 2. Opis projektu

Projekt predstavuje webovu aplikaciu vytvorenu v prostredi Python, ktora umoznuje trenovanie modelov pre casove rady nad historickymi datami akcii.

Hlavne ciele projektu:
- navrhnut a implementovat pipeline pre nacitanie, pripravu a trenovanie dat,
- porovnat realne a predikovane hodnoty ceny akcie,
- vizualizovat priebeh trenovania a kvalitu modelu,
- realizovat kratkodobu predikciu buducich hodnot.

Aktualne implementovane modely:
- LSTM
- Transformer (volba `trs`)

Hlavna aplikacna vrstva je v subore `app.py`, treningovy proces je implementovany v `training_4_0.py`.

### Pouzite technologie

- Python 3.11
- PyTorch
- Dash
- Plotly
- NumPy
- Pandas
- scikit-learn
- TensorBoard
- yfinance
- requests
- SQLAlchemy
- SQLite
- TensorFlow

## 3. Struktura projektu

- `app.py` - webove rozhranie, konfiguracia treningu, zobrazenie metrik a grafov
- `training_4_0.py` - hlavny treningovy pipeline
- `helper_functions/` - nacitanie dat, definicia modelov, predikcia, ukladanie vystupov, TensorBoard setup
- `database_scripts/` - skripty pre nacitanie a aktualizaciu dat do SQLite
- `api/api_functions.py` - sprava Alpha Vantage API klucov
- `models/` - ulozene modely (checkpointy)
- `runs/` - TensorBoard logy pre trening a predikciu
- `assets/style.css` - stylovanie Dash aplikacie

## 4. Instalacia a spustenie

### 4.1 Poziadavky

- Python 3.11 alebo novsi
- pip

### 4.2 Klon repozitara a vytvorenie virtualneho prostredia

```bash
git clone https://github.com/dragox134/stocks_data_prediction.git
cd stocks_data_prediction
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

### 4.3 Instalacia zavislosti

```bash
pip install -r requirements.txt
```

### 4.4 Spustenie aplikacie

```bash
python app.py
```

Po spusteni je aplikacia dostupna na adrese:
- `http://127.0.0.1:8050`

### 4.5 Volitelne: aktualizacia lokalnych databaz

Pre pracu s lokalnymi SQLite databazami su pripravene skripty:
- `database_scripts/start.py`
- `database_scripts/load_and_update.py`
- `api/api_functions.py`

Pred pouzitim je potrebne pripravit databazu API klucov (`dbs/api/api_keys.db`) s tabulkou `api_keys`.

Spustenie:

```bash
python database_scripts/start.py
```

Poznamka: Trening modelov v aktualnej verzii nacitava historicke data primarne cez kniznicu `yfinance`.

## 5. Zakladny postup pouzitia

1. V aplikacii vyber ticker, model a treningove parametre.
2. Spusti trening tlacidlom `Train Model`.
3. Sleduj vystupy:
- `Loss/Train`
- `Loss/Val`
- `Train/Close`
- `Test/Close` (vratane buducej predikcie)

## 6. Credits

- Autor projektu: Leo Gürtner
- Typ projektu: skolsky maturitny projekt
- Pouzite kniznice a frameworky: PyTorch, Dash, Plotly, scikit-learn, yfinance, SQLAlchemy, TensorBoard
- Data zdroje: Yahoo Finance (cez `yfinance`), Alpha Vantage (pre databazove skripty)

## 7. Prispievanie

Projekt je primarne urceny ako skolsky projekt. Pripadne navrhy na zlepsenie su vitane.

Odporucany postup:
1. Fork repozitara
2. Vytvorenie vlastneho branchu
3. Vytvorenie Pull Requestu s popisom zmien

## 8. Licencia

Tento projekt je licencovany pod MIT licenciou.
Podrobnosti su uvedene v subore `LICENSE`.
