import numpy as np
import os

# --- Configuration ---
INPUT_PATH = "../dataset/mimic-ecg.npy"
OUTPUT_PATH = "../dataset/mimic-ecg-small.npy"
NUM_SAMPLES = 5000  # 5000 samples is enough for a quick test

def create_subset():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: File not found at {INPUT_PATH}")
        return

    print(f"Reading {INPUT_PATH} in read-only mode (mmap)...")
    
    # mmap_mode='r' creates a virtual link to the file without loading it to RAM
    # This is instant and uses almost 0 memory.
    full_data = np.load(INPUT_PATH, mmap_mode='r')
    
    print(f"Full Dataset Shape: {full_data.shape}")
    
    # Create the slice (this still doesn't load data yet)
    small_data = full_data[:NUM_SAMPLES]
    
    print(f"Creating small dataset with shape: {small_data.shape}")
    print(f"Saving to {OUTPUT_PATH}...")
    
    # This effectively copies the first NUM_SAMPLES to the new file
    np.save(OUTPUT_PATH, small_data)
    
    print("✅ Done! You can now run your test.")

if __name__ == "__main__":
    create_subset()