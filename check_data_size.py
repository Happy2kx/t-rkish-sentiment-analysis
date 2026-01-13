import pandas as pd
import os

# Yolu tanımla
file_path = os.path.join("data", "raw", "turkish_sentiment_data.csv")

if os.path.exists(file_path):
    print(f"Reading {file_path}...")
    try:
        df = pd.read_csv(file_path)
        
        # Sütunları kontrol et
        print(f"Columns: {df.columns.tolist()}")
        
        # Sütun adlarını standartlaştırma mantığı (simüle edilmiş)
        text_col = None
        label_col = None
        for col in df.columns:
            lower_col = col.lower()
            if lower_col in ['text', 'content', 'review', 'comment']:
                text_col = col
            elif lower_col in ['label', 'sentiment', 'score']:
                label_col = col
                
        if text_col and label_col:
            print(f"Using text column: {text_col}, label column: {label_col}")
            df = df.dropna(subset=[text_col, label_col])
            
            # Toplam ham örnek
            total_samples = len(df)
            print(f"Total Raw Samples: {total_samples}")
            
            # Sınıf dağılımı
            distribution = df[label_col].value_counts()
            print("\nClass Distribution:")
            print(distribution)
            
            # Alt-örnek hesaplama
            min_count = distribution.min()
            num_classes = len(distribution)
            balanced_total = min_count * num_classes
            
            print(f"\nMinimum Class Count: {min_count}")
            print(f"Number of Classes: {num_classes}")
            print(f"Calculated Balanced Total: {balanced_total}")
            
        else:
            print("Text or Label column not found.")
            
    except Exception as e:
        print(f"Error reading CSV: {e}")
else:
    print(f"File not found: {file_path}")
