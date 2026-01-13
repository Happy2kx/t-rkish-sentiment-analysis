from PIL import Image
import sys
import os

def convert_to_ico(source_path, target_path):
    try:
        img = Image.open(source_path)
        # Gerekirse yeniden boyutlandır, ancak ico birden fazla boyutu destekler
        icon_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
        img.save(target_path, format='ICO', sizes=icon_sizes)
        print(f"Successfully converted {source_path} to {target_path}")
    except Exception as e:
        print(f"Error converting image: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_icon.py <source_png> <target_ico>")
    else:
        convert_to_ico(sys.argv[1], sys.argv[2])
