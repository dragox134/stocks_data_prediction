# Predikcia cien akcií na základe historických dát

Maturitný projekt zameraný na predikciu cien akcií pomocou modelov strojového učenia.

## Opis projektu

Projekt predstavuje webovú aplikáciu vytvorenú v prostredí Python, ktorá umožňuje
trénovanie modelov pre časové rady pomocou historických dát akcií.

Hlavné ciele projektu:
- navrhnúť a implementovať pipeline pre načítanie, prípravu a trénovanie dát,
- porovnať reálne a predikované hodnoty ceny akcie,
- vizualizovať priebeh trénovania a kvalitu modelu,
- realizovať krátkodobú predikciu budúcich hodnôt.

Aktuálne implementované modely:
- LSTM
- Transformer (voľba `trs`)

### Použité technológie

- Python 3.11
- PyTorch
- Dash
- Plotly
- NumPy
- Pandas
- scikit-learn
- TensorBoard
- yfinance

## Štruktúra projektu

- `app.py` — webové rozhranie, konfigurácia tréningu, zobrazenie metrík a grafov
- `training_4_0.py` — hlavný tréningový pipeline
- `helper_functions/` — načítanie dát, definícia modelov, predikcia, ukladanie výstupov, TensorBoard setup
- `assets/style.css` — štýlovanie Dash aplikácie
- `archived` — niekoľko hlavných uložených spustení
- `logs_pytorch` — všetky uložené spustenia
- `models` — všetky uložené najlepšie modely

> **Poznámka:** Súčasťou repozitára sú aj staršie skripty pre prácu s lokálnymi
> SQLite databázami a Alpha Vantage API (`database_scripts/`, `dbs/`, `api/`).
> Tieto komponenty vznikli ako prototyp v ranej fáze vývoja a nie sú súčasťou
> hlavnej funkcionality aplikácie — slúžia výlučne ako ukážka počiatočného smeru projektu.
> Aplikácia v aktuálnej verzii načítava dáta výhradne cez knižnicu `yfinance`.

## Inštalácia a spustenie

### Požiadavky

- Python 3.11 alebo novší
- pip

### Klon repozitára a vytvorenie virtuálneho prostredia

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

### Inštalácia závislostí

```bash
pip install -r requirements.txt
```

### Spustenie aplikácie

```bash
python app.py
```

Po spustení je aplikácia dostupná na adrese `http://127.0.0.1:8050`.

## Základný postup použitia

1. V aplikácii vyberte model a tréningové parametre.
2. Spustite tréning tlačidlom `Train Model`.
3. Sledujte výstupy:
   - `Loss/Train`
   - `Loss/Val`
   - `Train/Close`
   - `Test/Close` (vrátane budúcej predikcie)

## Credits

- **Autor projektu:** Leo Gürtner
- **Vedúci práce:** Ing. Oliver Halaš
- **Zdroje dát:** Yahoo Finance (cez `yfinance`)

## Prispievanie

Projekt je primárne určený ako školský projekt. Prípadné návrhy na zlepšenie sú vítané.

Odporúčaný postup:
1. Fork repozitára
2. Vytvorenie vlastného branchu
3. Vytvorenie Pull Requestu s popisom zmien

## Licencia

Tento projekt je licencovaný pod MIT licenciou.
Podrobnosti sú uvedené v súbore `LICENSE`.
