##  Повышение эффективности языковых моделей для детектирования аномалий в промышленных системах управления

Этот репозиторий содержит код выпускной квалификационной работы Степашкиной В. П. **«Разработка подхода к повышению эффективности языковых моделей для детектирования аномалий в промышленных системах управления»**.

---

### 📦 Установка

Клонируйте репозиторий и установите зависимости из `requirements.txt`. Мы рекомендуем использовать Python 3.9 и Pytorch 2.1 с поддержкой CUDA.

```bash
git clone https://github.com/imunellka/ICS-anomaly-detection-SWAT-WADI.git
cd ICS-anomaly-detection-SWAT-WADI

conda create --name efficientlm python=3.8
conda activate efficientlm

pip install torch==2.1.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

В проекте использовались 2 публичных датасета SWaT и WADI.
Посмотреть и скачать датасеты можно, запросив права у ITrust lab.

```
DATASET_DIR = 'path/to/dataset/processed/'
```

## 📁 Структура репозитория

Данный репозиторий содержит все необходимые компоненты для обучения, тестирования и демонстрации моделей аномалий в системе водоснабжения на основе различных архитектур.

### 📂 `data/`
- Информация о том, где взять исходные датасеты (SWaT и WADI).
- Пользовательский класс `WaterSystemDataset` для загрузки и обработки данных.

### 📂 `demo/`
- Все файлы, необходимые для запуска **веб-приложения на Streamlit**.

### 📂 `model_storage/`
- Предобученные веса моделей, используемые для инференса в демо-режиме.

### 📂 `models/`
- Код различных архитектур моделей:
  - AutoEncoderCN.py
  - TransformerbasedEncoder.py
  - ProbabilisticTranEncoder.py
  - MaskedTranEncoder.py
  - WaterSystemAnomalyTrasformer.py

### 📂 `training/`
- Скрипты для обучения моделей.
- Включает настройку тренировки, логирование и сохранение результатов.

### 📂 `utils/`
- Вспомогательные методы:
  - Разделение данных на окна (`get_windows`)
  - Препроцессинг и нормализация
  - Загрузка и подготовка данных

### 📄 `compute_metrics.py`
- Содержит методы для вычисления **оптимальных метрик качества**:
  - ROC AUC
  - F1-score
  - Подбор порогов + Point adjustment
