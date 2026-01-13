import matplotlib.pyplot as plt
import os
import re

# Çıktı dizini
output_dir = r"C:\Users\enest\.gemini\antigravity\brain\e568cdf7-a23b-4029-b44e-f01616e19255"

def save_code_image(code_text, filename, title=""):
    # Koyu tema ayarları
    bg_color = "#1e1e1e"  # VS Code Dark background
    text_color = "#d4d4d4"
    keyword_color = "#569cd6"
    string_color = "#ce9178"
    comment_color = "#6a9955"
    func_color = "#dcdcaa"
    
    # Şekil oluştur
    fig = plt.figure(figsize=(12, 12), facecolor=bg_color)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.set_facecolor(bg_color)
    ax.axis('off')
    
    # Basit sözdizimi vurgulama (regex tabanlı)
    # Bu, kodun kod gibi görünmesini sağlamak için temel bir simülasyondur
    # Mümkünse metni parçalara ayırıp parça parça renklendireceğiz,
    # ancak Matplotlib metni bir dizede çok renkli desteklemiyor.
    # O yüzden "kesin kodu" sağlamak için güzel bir monospace font ve beyaz metin kullanacağız.
    # Kullanıcı "sadece kod olacak şekilde" ve "birebir aynı" dedi.
    # Düz metin, bozuk karışık renklendirmeden daha güvenlidir.
    
    # BUT we can do basic keyword highlighting by using regex to identify keywords
    # and rendering them separately? No, that's too complex for mpl text alignment.
    # We will stick to a clean, high-quality white/grey monospace render.
    
    font_prop = {'family': 'consolas', 'size': 10, 'color': text_color}
    
    # Metni tek blok olarak işleyeceğiz
    ax.text(0, 1, code_text,
            transform=ax.transAxes,
            fontsize=11,
            color=text_color,
            family='monospace',
            verticalalignment='top',
            bbox=dict(facecolor=bg_color, alpha=0, edgecolor='none'))
            
    # Kaydet
    plt.savefig(os.path.join(output_dir, filename), dpi=300, facecolor=bg_color, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Created {filename}")

# 1. Veri Yükleme (src/data_preprocessing.py)
code_1 = r'''def load_dataset(file_path):
    """
    Load dataset from CSV file.
    Expected format: 'text' and 'label' columns.
    If columns are different, it attempts to rename them.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Standardize column names
    text_col = None
    label_col = None
    
    for col in df.columns:
        lower_col = col.lower()
        if lower_col in ['text', 'content', 'review', 'comment', 'tweet']:
            text_col = col
        elif lower_col in ['label', 'sentiment', 'score', 'target']:
            label_col = col
            
    if text_col and label_col:
        df = df.rename(columns={text_col: 'text', label_col: 'label'})
        df = df[['text', 'label']]
    else:
        print(f"Warning: Could not automatically detect text/label columns. Available columns: {df.columns}")
        
    df = df.dropna(subset=['text', 'label'])
    
    return df'''

save_code_image(code_1, "ekran_1_veri_yukleme.png")

# 2. Metin Temizleme (src/data_preprocessing.py)
code_2 = r'''def clean_text(text):
    """
    Basic text cleaning:
    - Lowercase
    - Remove punctuation and numbers
    - Remove extra whitespace
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text'''

save_code_image(code_2, "ekran_2_metin_temizleme.png")

# 3. TF-IDF (src/data_preprocessing.py)
code_3 = r'''def get_vectorizer(max_features=10000, ngram_range=(1, 2)):
    """
    Create and return a TfidfVectorizer with N-gram support.
    
    N-gram (1,2) captures:
    - Unigrams: Individual words ("güzel", "kötü")
    - Bigrams: Word pairs ("çok güzel", "güzel değil")
    
    This helps capture negation and context: "güzel değil" -> Negative
    """
    return TfidfVectorizer(
        max_features=max_features, 
        ngram_range=ngram_range,  # Unigram + Bigram
        min_df=3,                  # Word must appear in at least 3 docs
        max_df=0.9                 # Ignore words in >90% of docs
    )'''

save_code_image(code_3, "ekran_3_tfidf.png")

# 4. Veri Dengeleme (src/data_preprocessing.py)
code_4 = r'''def balance_dataset(df, random_state=42):
    """
    Balance the dataset by undersampling majority classes.
    This ensures each class has the same number of samples as the minority class.
    """
    print("\nBalancing dataset...")
    print(f"Original distribution:\n{df['label'].value_counts()}")
    
    # Find the minimum class count
    min_count = df['label'].value_counts().min()
    print(f"Minimum class count: {min_count}")
    
    # Undersample each class to match the minority class
    balanced_dfs = []
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        if len(label_df) > min_count:
            label_df = label_df.sample(n=min_count, random_state=random_state)
        balanced_dfs.append(label_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle the balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"Balanced distribution:\n{balanced_df['label'].value_counts()}")
    
    return balanced_df'''

save_code_image(code_4, "ekran_4_dengeleme.png")

# 5. Grid Search (src/model_trainer.py)
code_5 = r'''def train_with_grid_search(model, param_grid, X_train, y_train, model_name, cv=3):
    """
    Train a model with Grid Search and Stratified K-Fold Cross Validation.
    """
    print("\n[GRID SEARCH] Training {} with Grid Search ({}-Fold CV)...".format(model_name, cv))
    print("   Parameter grid: {}".format(param_grid))
    
    start_time = time.time()
    
    # Stratified K-Fold ensures each fold has same class distribution
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Grid Search with Cross Validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=skf,
        scoring='f1_weighted',  # F1 score for imbalanced data
        n_jobs=-1,              # Use all CPU cores
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    return grid_search.best_estimator_, train_time, grid_search.best_params_, grid_search.best_score_'''

save_code_image(code_5, "ekran_5_grid_search.png")

# 6. Ensemble (src/model_trainer.py)
code_6 = r'''def train_voting_ensemble(models_dict, X_train, y_train):
    """
    Create and train a Voting Ensemble from trained models.
    """
    print("\n[ENSEMBLE] Creating Voting Ensemble...")
    start_time = time.time()
    
    # Prepare estimators list for VotingClassifier
    estimators = []
    
    for name, model in models_dict.items():
        if isinstance(model, LinearSVC):
            calibrated = CalibratedClassifierCV(model, cv=3, method='sigmoid')
            calibrated.fit(X_train, y_train)
            estimators.append((name, calibrated))
        else:
            estimators.append((name, model))
    
    # Create Voting Classifier with soft voting (probability-based)
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',  # Use probability averaging
        n_jobs=-1
    )
    
    ensemble.fit(X_train, y_train)
    return ensemble, time.time() - start_time'''

save_code_image(code_6, "ekran_6_ensemble.png")
