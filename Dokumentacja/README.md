 🏀 NBA Game Outcome Prediction System

System predykcji wyników meczów NBA wykorzystujący zaawansowane algorytmy machine learning oraz analizę reguł asocjacyjnych.

## 📊 Opis Projektu

Projekt analizuje ponad **30,000 meczów NBA** z lat 1946-2023, wykorzystując dane historyczne do przewidywania wyników przyszłych spotkań. System implementuje 8 różnych algorytmów ML oraz odkrywa ukryte wzorce w danych przy użyciu reguł asocjacyjnych.

## 🎯 Cele Projektu

- **Predykcja wyników** meczów NBA na podstawie historycznych statystyk drużyn
- **Porównanie skuteczności** różnych algorytmów machine learning
- **Odkrycie wzorców** w danych sportowych przy użyciu reguł asocjacyjnych
- **Analiza czynników** wpływających na wyniki meczów (forma drużyny, przewaga gospodarzy)

## 🏗️ Architektura Systemu

```
MLProjekt/
├── 📊 Dane
│   ├── data_loader.py          # Ładowanie danych NBA
│   └── data_preprocessing.py   # Przygotowanie cech predykcyjnych
├── 🤖 Modele ML
│   ├── model_training.py       # Trenowanie 8 algorytmów
│   ├── hyperparameter_tuning.py # Optymalizacja parametrów
│   └── decision_tree_analysis.py # Analiza drzew decyzyjnych
├── 🔍 Analiza
│   ├── association_rules.py    # Reguły asocjacyjne (Apriori)
│   ├── error_analysis.py       # Analiza błędów modeli
│   └── visualizations.py       # Wizualizacje i wykresy
├── 📋 Wyniki
│   └── wyniki/                 # Raporty, wykresy, modele
└── main.py                     # Główny pipeline
```

## 🚀 Szybki Start

### Wymagania
```bash
pip install pandas numpy scikit-learn matplotlib seaborn mlxtend
```
i pobranie bazy danych z Kaggle (nie chcialem miec na repo bazy 5GB)



### Pobranie danych
1. Pobierz [NBA Games Dataset z Kaggle](https://www.kaggle.com/datasets/wyattowalsh/basketball)
2. Rozpakuj do folderu `csv/`
3. Struktura powinna być:
   ```
   csv/
   ├── games.csv
   ├── teams.csv
   └── ...
   ```
### Uruchomienie
```bash
python main.py
albo python =m MLProjekt.main
```

## 📈 Wyniki

### 🏆 Najlepsze Modele
| Model | Dokładność | Precision | Recall | F1-Score |
|-------|------------|-----------|--------|----------|
| **Naive Bayes** | **60.90%** | 0.65 | 0.73 | 0.69 |
| Gradient Boosting | 59.87% | 0.66 | 0.62 | 0.67 |
| Random Forest | 59.00% | 0.64 | 0.62 | 0.66 |
| Logistic Regression | 58.45% | 0.63 | 0.64 | 0.63 |

### 🔍 Kluczowe Odkrycia

1. **Forma drużyny** (`win_pct_diff`) - najważniejszy predyktor (90% ważności w drzewie)
2. **Przewaga gospodarzy** - występuje w większości reguł asocjacyjnych
3. **Proste modele** przewyższają złożone sieci neuronowe
4. **Naturalna granica** dokładności sportowej (~61%) ze względu na losowość

### 📊 Reguły Asocjacyjne
- **7,185 reguł** odkrytych algorytmem Apriori
- **Najsilniejsza reguła**: `[home_adv=wysoki, fg_pct_diff=wysoki] → [win_pct_diff=wysoki]`
  - Lift: 2.45, Confidence: 80.6%

## 🔬 Metodologia

### 1. Przygotowanie Danych
- **Cechy predykcyjne**: 10 różnic statystycznych między drużynami
- **Okno czasowe**: Analiza 10 poprzednich meczów każdej drużyny
- **Podział danych**: 80% trening / 20% test (chronologicznie)

### 2. Modele ML
- Naive Bayes Gaussian
- Random Forest (50-500 drzew)
- Gradient Boosting
- Neural Networks (MLP, Deep NN)
- Logistic Regression
- K-Nearest Neighbors
- Decision Trees

### 3. Ewaluacja
- **Cross-validation** 5-fold
- **Metryki**: Accuracy, Precision, Recall, F1-Score, AUC
- **Wizualizacje**: Confusion matrices, ROC curves, Learning curves

## Technologie

- **Python 3.8+**
- **Scikit-learn**: Modele ML i ewaluacja
- **Pandas/NumPy**: Przetwarzanie danych
- **Matplotlib/Seaborn**: Wizualizacje
- **MLxtend**: Reguły asocjacyjne (Apriori)
- **Joblib**: Serializacja modeli

##  Źródła Danych

- **NBA Games Dataset** (Kaggle)
- **Okres**: 1946-2023
- **Rozmiar**: 30,000+ meczów
- **Cechy**: Statystyki drużyn, wyniki, daty


**Projekt Machine Learning - Semestr 4**  
Analiza predykcji wyników NBA

## 📄 Licencja

Projekt edukacyjny - wykorzystanie zgodnie z polityką uczelni.
