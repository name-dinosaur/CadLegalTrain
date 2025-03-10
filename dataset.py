import pandas as pd
from datasets import load_dataset

dataset = load_dataset("refugee-law-lab/canadian-legal-data")
dataset.save_to_disk("canadian_legal_data")
