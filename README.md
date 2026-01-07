MULTIMODAL HOUSE PRICE PREDICTION
Integrating Tabular Data and Satellite Imagery for Real Estate Valuation


OVERVIEW

Traditional house price prediction models rely mainly on structured data such as size, location, and amenities. While effective, they fail to capture the visual and environmental context surrounding a property — greenery, road density, water proximity, and neighborhood layout — all of which strongly influence market value.

This project builds a Multimodal Regression Pipeline that predicts property prices by fusing:

Tabular data (numerical and categorical property attributes)

Satellite imagery (environmental context from overhead views)

By learning from both numbers and pixels, the system produces more realistic and robust property valuations.



OBJECTIVES

Build a multimodal regression model for house price prediction

Programmatically acquire satellite images using latitude and longitude

Extract visual embeddings using CNNs

Learn numerical embeddings using a deep tabular network

Fuse both modalities into a unified prediction model

Compare:

Tabular-only baseline

Image-based signal

Multimodal fusion

Create a scalable pipeline for future real-estate analytics use cases



THOUGHT PROCESS

Property valuation is driven by two parallel information streams:

Quantitative signals – size, rooms, age, location statistics

Qualitative signals – greenery, density, road networks, surroundings

Instead of manually engineering image features, this system:

Learns representations automatically

Keeps each modality independent during feature learning

Fuses them only after high-level abstraction



SYSTEM ARCHITECTURE

High-level flow:

Raw Data
-> Tabular CSV -> Tabular Encoder -> Tabular Embeddings
-> Lat/Long -> Google Maps API -> Satellite Images -> CNN Encoder -> Image Embeddings
-> Fusion Network -> Final Price Prediction



SETUP INSTRUCTIONS

Clone the repository

git clone https://github.com/abhayrajawat314/House-Price-Prediction.git

cd House-Price-Prediction

Create a virtual environment

python -m venv venv
source venv/bin/activate (Linux / Mac)
venv\Scripts\activate (Windows)

Install dependencies

pip install -r requirements.txt

Configure Google Maps API key

Linux / Mac:
export GOOGLE_MAPS_API_KEY="YOUR_KEY"

Windows:
setx GOOGLE_MAPS_API_KEY "YOUR_KEY"

HOW TO RUN THE PROJECT

Step 1 — Download satellite images
python src/Components/data_fetcher.py

Step 2 — Train tabular encoder
python train_tabular_encoder.py

Creates:
artifacts/tabular_encoder.keras
artifacts/tabular_scaler.pkl
artifacts/tabular_features.pkl

Step 3 — Train satellite CNN encoder
python train_satellite_encoder.py

Creates:
artifacts/satellite_encoder.keras

Step 4 — Train tabular-only baseline (optional)
python train_tabular_only_model.py

Creates:
artifacts/tabular_only_model.keras

Step 5 — Train multimodal fusion model
python train_fusion_model.py

Creates:
artifacts/fusion_model.keras

Step 6 — Run inference
python predict_fusion_model.py

Creates:
artifacts/test_predictions.csv

WHAT THE MODEL LEARNS

From tabular data:

Property size

Amenities

Locality statistics

From satellite images:

Green cover

Road connectivity

Neighborhood density

From fusion:

Non-linear interaction between both data types



EXPERIMENTS AND OBSERVATIONS

Tabular-only model

Strong baseline

Misses environmental influence

Image signal alone

Higher variance

Needs fusion for stability

Multimodal fusion

Best generalization

Learns neighborhood impact on prices

Reduces overfitting to any single modality



AUTHOR

Abhay Rajawat
