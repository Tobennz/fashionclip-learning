# FashionCLIP Product Recommender

A multimodal fashion recommendation application built with Python, FashionCLIP, FastAPI, and vector similarity search.

Users can search for similar fashion products using:

- A text description
- An uploaded product image
- Text and an image together

The application converts products and search queries into vectors, compares them using cosine similarity, and returns the closest matching products.

## Features

- Text-based fashion product search
- Image-based fashion product search
- Combined image and text search
- FashionCLIP image and text embeddings
- Vector normalization and fusion
- Cosine similarity ranking
- Preselectable number of recommendations
- Precomputed catalog index for faster searches
- FastAPI backend
- Interactive Swagger documentation
- Browser-based search interface
- Automated tests with pytest

## How It Works

1. Product images and descriptions are converted into vectors using FashionCLIP.
2. Each image vector and text vector is normalized.
3. The vectors are joined to create one multimodal product vector.
4. Catalog vectors are saved so they do not need to be recreated for every search.
5. The user's text, uploaded image, or both are converted into a query vector.
6. Cosine similarity compares the query vector with the saved catalog vectors.
7. The products with the highest similarity scores are returned as recommendations.

## Project Structure

```text
fashionclip-learning/
├── configs/
│   └── base.yaml
├── data/
│   ├── images/
│   └── products.csv
├── frontend/
│   └── index.html
├── scripts/
│   ├── build_catalog_index.py
│   ├── check_image_text.py
│   ├── check_real_model.py
│   ├── recommend_real_products.py
│   └── search_saved_index.py
├── src/
│   ├── app/
│   │   ├── api.py
│   │   ├── index_store.py
│   │   ├── inference.py
│   │   └── recommender.py
│   ├── data/
│   │   ├── fashion_dataset.py
│   │   ├── product_loader.py
│   │   └── text_template.py
│   ├── embedders/
│   │   └── fashionclip_embedder.py
│   └── pipeline/
│       ├── fusion.py
│       └── similarity.py
├── tests/
├── main.py
├── pyproject.toml
└── README.md
```

## Installation

This project uses Python 3.12 and `uv` for dependency management.

Clone the repository and enter the project folder:

```powershell
git clone https://github.com/Tobennz/fashionclip-learning.git
cd fashionclip-learning
```

Install the dependencies:

```powershell
uv sync
```

The first time FashionCLIP runs, its model files will be downloaded from Hugging Face.

## Dataset Setup

This project uses the Fashion Product Images dataset from Kaggle:

https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

Download and extract the dataset under:

```text
data/fashion/
```

The downloaded dataset is excluded from Git because it contains thousands of images.

## Build the Catalog Index

Create the saved FashionCLIP vector index:

```powershell
uv run python -m scripts.build_catalog_index
```

This loads the catalog, creates image and text embeddings, combines them into multimodal vectors, and saves the result under:

```text
outputs/catalog/
```

The generated index is excluded from Git because it can be rebuilt from the dataset.

## Run the Application

Start the FastAPI server:

```powershell
uv run uvicorn src.app.api:app --reload
```

Open the visual interface:

```text
http://127.0.0.1:8000/ui
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Stop the server by pressing `Ctrl+C` in the terminal.

## Run the Tests

Run the complete automated test suite:

```powershell
uv run pytest -v --basetemp=.pytest-temp
```

The tests cover dataset loading, embeddings, vector fusion, similarity search, saved indexes, inference, API routes, and image uploads.

## API Usage

### Text Search

Send a JSON request to:

```text
POST /recommend
```

Example request:

```json
{
  "text": "navy blue checked men's shirt",
  "k": 5
}
```

### Image and Text Search

Send a multipart form request to:

```text
POST /recommend/upload
```

The form accepts:

- `image`: an optional image file
- `text`: an optional description
- `k`: the number of recommendations

At least one of `image` or `text` must be provided.

### Product Images

Recommended product images are served through:

```text
GET /images/{filename}
```

Example:

```text
http://127.0.0.1:8000/images/15970.jpg
```

## Technologies Used

- Python 3.12
- FashionCLIP
- PyTorch
- Hugging Face Transformers
- NumPy
- Pillow
- FastAPI
- Uvicorn
- Pytest
- HTML, CSS, and JavaScript
- uv

## Data and Model Attribution

Product metadata and images come from the Fashion Product Images dataset:

https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

This project uses the FashionCLIP model:

https://huggingface.co/patrickjohncyh/fashion-clip

FashionCLIP and the dataset remain subject to their respective licenses and terms of use.

## Project Status

This is a learning project demonstrating an end-to-end multimodal machine-learning application. It includes data loading, embedding generation, vector search, an API, automated tests, and a browser interface.