# From Model to Production — Fashion Product Classification

This project trains and deploys a model to classify fashion product images. It includes automation using GitHub Actions.

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate   # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small?select=styles.csv).

Place files as follows:

```
Data/
├── styles.csv
└── training_images/
    └── (image files)
```

4. Train the model:

```bash
python training.py
```

5. Run the app:

```bash
python app.py
```

## GitHub Actions

- Workflow files are in `.github/workflows/`
- `nightly_prediction.yml` runs batch predictions daily at 2:00 AM

## License

MIT License
