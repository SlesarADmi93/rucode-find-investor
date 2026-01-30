# test_gen.py
from src.data import generate_synthetic_data

df = generate_synthetic_data(n_samples=100, seed=42)
print(df.head())
print("Shape:", df.shape)
print("Accepted ratio:", df["accepted"].mean())
