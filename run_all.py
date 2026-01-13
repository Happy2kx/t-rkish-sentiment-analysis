import os
import pandas as pd
import pickle
from src.download_data import download_and_save_data
from src.data_preprocessing import load_dataset, preprocess_data, balance_dataset, create_train_test_split, get_vectorizer
from src.model_trainer import train_all_models
from src.model_evaluator import compare_models, generate_comparison_plots, create_confusion_matrix, evaluate_model


def train_and_evaluate(df, models_dir, dataset_type="balanced", use_grid_search=True, cv_folds=3):
    """
    Belirli bir veri seti üzerinde Grid Search ve CV ile tüm modelleri eğit ve değerlendir.
    
    Parametreler:
    -----------
    df : DataFrame
        Ön işlemden geçmiş veri seti
    models_dir : str
        Modelleri kaydetmek için dizin
    dataset_type : str
        Veri seti tipi ("balanced" veya "raw")
    use_grid_search : bool
        Grid Search optimizasyonunun kullanılıp kullanılmayacağı
    cv_folds : int
        Çapraz doğrulama fold sayısı
    """
    print("\n" + "="*60)
    print("   Training on {} dataset".format(dataset_type.upper()))
    print("   Grid Search: {}".format('Enabled' if use_grid_search else 'Disabled'))
    print("   CV Folds: {}".format(cv_folds))
    print("="*60)
    
    # Modeller dizinini oluştur
    os.makedirs(models_dir, exist_ok=True)
    
    # Veriyi böl
    X_train, X_test, y_train, y_test = create_train_test_split(df)
    print("\n[DATA] Data Split:")
    print("   Training set: {:,} samples".format(len(X_train)))
    print("   Test set: {:,} samples".format(len(X_test)))
    
    # N-gram (1,2) ile vektörleştirme
    print("\n[TEXT] Vectorizing text with N-gram (1,2)...")
    vectorizer = get_vectorizer()  # Varsayılan olarak ngram_range=(1,2) kullanır
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print("   Feature count: {:,} features".format(X_train_vec.shape[1]))
    
    # Grid Search ve CV ile Model Eğitimi
    results = train_all_models(
        X_train_vec, y_train, vectorizer, models_dir,
        use_grid_search=use_grid_search, 
        cv=cv_folds
    )
    
    # Test setinde değerlendirme
    print("\n" + "="*60)
    print("   EVALUATING MODELS ON TEST SET")
    print("="*60)
    
    loaded_models = {}
    for name in results.keys():
        model_path = os.path.join(models_dir, "{}.pkl".format(name))
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                loaded_models[name] = pickle.load(f)
            
    comparison_df = compare_models(loaded_models, X_test_vec, y_test)
    
    # Eğitim sürelerini ve CV skorlarını ekle
    for name in results.keys():
        if name in comparison_df['Model'].values:
            comparison_df.loc[comparison_df['Model'] == name, 'Training Time (s)'] = results[name]['time']
            comparison_df.loc[comparison_df['Model'] == name, 'CV Score'] = results[name].get('cv_score', 0.0)
    
    # Veri seti tipi sütunu ekle
    comparison_df['Dataset'] = dataset_type
    comparison_df['Grid Search'] = use_grid_search
    comparison_df['CV Folds'] = cv_folds
        
    print("\n[RESULTS] {} Results:".format(dataset_type.upper()))
    print("-" * 80)
    print(comparison_df[['Model', 'Accuracy', 'F1 Score', 'CV Score', 'Training Time (s)']].to_string(index=False))
    print("-" * 80)
    
    # En iyi modelleri vurgula
    best_accuracy = comparison_df['Accuracy'].max()
    best_f1 = comparison_df['F1 Score'].max()
    best_acc_model = comparison_df.loc[comparison_df['Accuracy'] == best_accuracy, 'Model'].values[0]
    best_f1_model = comparison_df.loc[comparison_df['F1 Score'] == best_f1, 'Model'].values[0]
    
    print("\n[BEST] Best Accuracy: {:.4f} ({})".format(best_accuracy, best_acc_model))
    print("[BEST] Best F1 Score: {:.4f} ({})".format(best_f1, best_f1_model))
    
    # Sonuçları kaydet
    comparison_df.to_csv(os.path.join(models_dir, 'comparison_results.csv'), index=False)
    
    # Grafikler oluştur
    generate_comparison_plots(comparison_df, models_dir)
    
    # Confusion matrix'ler
    for name, model in loaded_models.items():
        _, y_pred = evaluate_model(model, X_test_vec, y_test)
        create_confusion_matrix(y_test, y_pred, name, models_dir)
    
    return comparison_df


def main():
    print("=" * 70)
    print("   ENHANCED MODEL TRAINING PIPELINE")
    print("   Strategy 1: N-gram (1,2) + Grid Search + 3-Fold CV + Ensemble")
    print("=" * 70)
    
    # Yapılandırma
    USE_GRID_SEARCH = True
    CV_FOLDS = 3  # Stratified 3-Fold Çapraz Doğrulama
    
    # 1. Veri İndirme
    print("\n" + "="*60)
    print("   STEP 1: DATA ACQUISITION")
    print("="*60)
    raw_data_path = os.path.join('data', 'raw', 'turkish_sentiment_data.csv')
    if not os.path.exists(raw_data_path):
        download_and_save_data()
    else:
        print("[OK] Data already exists.")

    # 2. Ön İşleme
    print("\n" + "="*60)
    print("   STEP 2: DATA PREPROCESSING")
    print("="*60)
    df_raw = load_dataset(raw_data_path)
    df_preprocessed = preprocess_data(df_raw.copy())
    
    # Ham işlenmiş veriyi kaydet
    raw_processed_path = os.path.join('data', 'processed', 'processed_data_raw.csv')
    os.makedirs(os.path.dirname(raw_processed_path), exist_ok=True)
    df_preprocessed.to_csv(raw_processed_path, index=False)
    print("[OK] Raw processed data: {:,} samples".format(len(df_preprocessed)))
    
    # 3. Veri Setini Dengele
    print("\n" + "="*60)
    print("   STEP 3: DATA BALANCING")
    print("="*60)
    df_balanced = balance_dataset(df_preprocessed.copy())
    
    # Dengelenmiş veriyi kaydet
    balanced_path = os.path.join('data', 'processed', 'processed_data.csv')
    df_balanced.to_csv(balanced_path, index=False)
    print("[OK] Balanced data: {:,} samples".format(len(df_balanced)))
    
    # 4. HAM (dengesiz) veri üzerinde eğit
    print("\n" + "="*70)
    print("   PHASE 1: TRAINING ON RAW DATA")
    print("="*70)
    raw_results = train_and_evaluate(
        df_preprocessed, 
        models_dir='models_raw', 
        dataset_type="raw",
        use_grid_search=USE_GRID_SEARCH,
        cv_folds=CV_FOLDS
    )
    
    # 5. DENGELENMİŞ veri üzerinde eğit
    print("\n" + "="*70)
    print("   PHASE 2: TRAINING ON BALANCED DATA")
    print("="*70)
    balanced_results = train_and_evaluate(
        df_balanced, 
        models_dir='models_balanced', 
        dataset_type="balanced",
        use_grid_search=USE_GRID_SEARCH,
        cv_folds=CV_FOLDS
    )
    
    # 6. Nihai Özet
    print("\n" + "="*70)
    print("   TRAINING COMPLETE - FINAL SUMMARY")
    print("="*70)
    
    print("\n[RAW DATA MODELS] (models_raw/):")
    print("   Samples: {:,}".format(len(df_preprocessed)))
    print("   Best Accuracy: {:.2%}".format(raw_results['Accuracy'].max()))
    print("   Best F1 Score: {:.4f}".format(raw_results['F1 Score'].max()))
    
    print("\n[BALANCED DATA MODELS] (models_balanced/):")
    print("   Samples: {:,}".format(len(df_balanced)))
    print("   Best Accuracy: {:.2%}".format(balanced_results['Accuracy'].max()))
    print("   Best F1 Score: {:.4f}".format(balanced_results['F1 Score'].max()))
    
    # Ensemble vs Tekil modelleri karşılaştır
    print("\n[ENSEMBLE COMPARISON]:")
    for df, name in [(raw_results, "RAW"), (balanced_results, "BALANCED")]:
        ensemble_acc = df.loc[df['Model'] == 'Voting_Ensemble', 'Accuracy'].values
        if len(ensemble_acc) > 0:
            ensemble_acc = ensemble_acc[0]
            others_max = df.loc[df['Model'] != 'Voting_Ensemble', 'Accuracy'].max()
            diff = (ensemble_acc - others_max) * 100
            sign = '+' if diff >= 0 else ''
            print("   {}: Ensemble={:.4f}, Best Single={:.4f} ({}{:.2f}%)".format(
                name, ensemble_acc, others_max, sign, diff))
    
    print("\n" + "="*70)
    print("   STRATEGY 1 APPLIED SUCCESSFULLY!")
    print("   - N-gram (1,2) for context capture")
    print("   - Grid Search for hyperparameter optimization")
    print("   - Stratified 3-Fold CV for reliable validation")
    print("   - Voting Ensemble for model combination")
    print("="*70)
    print("\nRun 'streamlit run app.py' to view the interface.")


if __name__ == "__main__":
    if not os.path.exists('src'):
        print("[WARNING] Please run this script from the project root directory.")
    else:
        main()
