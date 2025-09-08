# Eye Disease Detection Project

This project demonstrates how to build, train, and deploy a deep‑learning model for eye‑disease classification.

## Quick Start

```bash
# 1. Create & activate a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Download and unzip your dataset into `data/`

# 4. Train the model
python model/train.py --dataset_dir data/ --epochs 20

# 5. Launch the web app
python main.py
```

The web interface will be available at **http://127.0.0.1:5000**.

## Folder Layout

```
.
├── app/              # Flask application
├── data/             # Place your raw images here (sub‑folders = class names)
├── model/            # Training scripts, saved weights, inference helpers
├── notebook/         # (Optional) Jupyter notebook version of train.py
├── utils/            # Shared utility functions
└── requirements.txt
```

## Dataset Format

```
data/
├── NORMAL/
│   ├── img_001.jpg
│   └── ...
├── CATARACT/
├── GLAUCOMA/
└── ...
```

Each sub‑folder represents a disease label.

## License

MIT