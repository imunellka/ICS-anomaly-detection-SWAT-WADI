##  Повышение эффективности языковых моделей для детектирования аномалий в промышленных системах управления

Этот репозиторий содержит код для проекта **«Разработка подхода к повышению эффективности языковых моделей для детектирования аномалий в промышленных системах управления»**.

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

В проекте использовались 2 публичных датасета SWaT, and WADI.
Посмотреть и скачать датасеты можно, запросив права у ITrust lab.

```
DATASET_DIR = 'path/to/dataset/processed/'
```
