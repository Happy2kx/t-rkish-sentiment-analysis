import pandas as pd
import matplotlib.pyplot as plt
import os

# Çıktı dizini
output_dir = r"C:\Users\enest\.gemini\antigravity\brain\e568cdf7-a23b-4029-b44e-f01616e19255"

def save_table_image(df, title, filename, col_widths=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Tablo oluştur
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    # Stil
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    # Başlık stili
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#404040')
        else:
            if row % 2 == 0:
                cell.set_facecolor('#f5f5f5')
    
    # Kaydet
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created {filename}")

# Tablo 3: Ham vs Dengeli
df3 = pd.DataFrame([
    ["Toplam Örnek", "440,679", "152,715", "-287,964"],
    ["Doğruluk (Accuracy)", "%92.89", "%89.88", "-3.01%"],
    ["F1 Skoru", "0.9283", "0.8984", "-0.0299"],
    ["Kesinlik (Precision)", "0.9280", "0.8991", "-0.0289"],
    ["Duyarlılık (Recall)", "0.9289", "0.8988", "-0.0301"],
    ["Eğitim Süresi", "233.77 sn", "128.44 sn", "-105.33 sn"]
], columns=["Metrik", "Ham Veri", "Dengeli Veri", "Fark"])

save_table_image(df3, "Tablo 3: Ham Veri ve Dengeli Veri Performans Karşılaştırması", "table_comparison_raw_balanced_py.png")

# Tablo 4: Algoritma Karşılaştırması
df4 = pd.DataFrame([
    ["Naive Bayes", "%87.56", "0.8754", "2.53 sn", "0.0012"],
    ["Logistic Regression", "%89.88", "0.8987", "8.12 sn", "0.0034"],
    ["SVM", "%89.76", "0.8975", "21.34 sn", "0.0089"],
    ["Random Forest", "%88.34", "0.8835", "89.67 sn", "0.0156"],
    ["Voting Ensemble", "%89.88", "0.8984", "6.78 sn", "0.0245"]
], columns=["Algoritma", "Doğruluk", "F1 Skoru", "Eğitim Süresi", "Tahmin Hızı (ms)"])

save_table_image(df4, "Tablo 4: Algoritma Performans Karşılaştırması", "table_algorithm_comparison_py.png")

# Tablo 5: Sınıf Bazlı Etki
df5 = pd.DataFrame([
    ["Pozitif", "0.94", "0.91", "0.96", "0.90"],
    ["Negatif", "0.93", "0.90", "0.91", "0.89"],
    ["Nötr", "0.87", "0.89", "0.84", "0.90"]
], columns=["Sınıf", "Ham Kesinlik", "Dengeli Kesinlik", "Ham Duyarlılık", "Dengeli Duyarlılık"])

save_table_image(df5, "Tablo 5: Sınıf Bazlı Veri Dengeleme Etkisi", "table_class_balance_effect_py.png")
