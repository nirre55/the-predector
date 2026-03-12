import argparse
import logging
import os

import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def main():
    parser = argparse.ArgumentParser(description='BTC Polymarket Predictor')
    parser.add_argument('mode', choices=['train', 'live'], help='Mode : train ou live')
    args = parser.parse_args()

    if args.mode == 'train':
        from data.downloader import download_historical
        from data.loader import load_data
        from features.engineering import compute_features
        from features.labeling import generate_labels
        from model.evaluator import evaluate
        from model.trainer import train_pipeline
        from config import DATA_PATH, MODEL_PATH, SCALER_PATH

        # Télécharger les données si absentes
        if not os.path.exists(DATA_PATH):
            print("Téléchargement des données historiques Binance...")
            download_historical()
        else:
            print(f"Données déjà présentes : {DATA_PATH}")

        print("Chargement des données...")
        df = load_data()
        print(f"  {len(df):,} bougies chargées ({df.index[0]} → {df.index[-1]})")

        print("Calcul des features...")
        features, names, feat_index = compute_features(df)
        print(f"  {features.shape[0]:,} exemples × {features.shape[1]} features")

        print("Génération des labels...")
        labels = generate_labels(df)

        # Aligner features et labels via l'index exact retourné par compute_features
        df_feat = pd.DataFrame(features, index=feat_index, columns=names)
        df_feat['y'] = labels.reindex(df_feat.index)
        df_feat = df_feat.dropna()

        X = df_feat.drop('y', axis=1).to_numpy(dtype=float)
        y = df_feat['y'].to_numpy(dtype=int)
        print(f"  Dataset final : {len(X):,} exemples")

        print("\nEntraînement du modèle...")
        train_pipeline(X, y)

        # Évaluation sur le dernier 20%
        print("\nÉvaluation sur le test set (20% final)...")
        n = len(X)
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        evaluate(model, scaler, X[int(0.8 * n):], y[int(0.8 * n):])

    elif args.mode == 'live':
        from live.engine import run_live
        print("Démarrage du mode live — en attente des données Binance WebSocket...")
        run_live()


if __name__ == '__main__':
    main()
