import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib


# Load the dataset and split into train/test
def load_data(file_path, target_col="Y", test_size=0.2, random_state=42):
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print("Full dataset distribution:")
    print(y.value_counts(normalize=True))
    print("\nTrain distribution:")
    print(y_train.value_counts(normalize=True))
    print("\nTest distribution:")
    print(y_test.value_counts(normalize=True))

    return df, X_train, X_test, y_train, y_test


# Explore categorical features
def plot_counts(df, features):
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 8))
    for i, f in enumerate(features, 1):
        plt.subplot(2, 3, i)
        sns.countplot(x=f, data=df)
        plt.title(f"{f} counts")
        # Add counts on top of bars
        for p in plt.gca().patches:
            height = p.get_height()
            if height > 0:
                plt.gca().annotate(int(height), (p.get_x() + p.get_width() / 2, height),
                                   ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

# Heatmap of correlations
def plot_corr(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

# Feature vs target plots
def feature_vs_target(df, features, target="Y"):
    plt.figure(figsize=(14, 8))

    for i, f in enumerate(features, 1):
        plt.subplot(2, 3, i)
        cross = pd.crosstab(df[f], df[target], normalize='index') * 100
        cross.plot(kind='bar', ax=plt.gca())
        plt.title(f"{f} vs {target}")
        plt.ylabel("Percent")
        plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# Simple feature importance using RandomForest
def get_feature_ranking(X, y):
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42
        )
    
    model.fit(X, y)

    imp_df = pd.DataFrame({
        "feature": X.columns, 
        "importance": model.feature_importances_
        })
    
    imp_df = imp_df.sort_values("importance", ascending=False)

    print("\nRandom Forest Feature Importance:")
    print(imp_df)
    return imp_df["feature"].tolist()


# Evaluate a model on test set and cross-validation
def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    if hasattr(model, "predict_proba"):
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        auc = None
    
    cv = cross_val_score(
        model, X_train, y_train, cv=5
        )

    return {
        "accuracy": acc, 
        "auc": auc, 
        "cv_mean": cv.mean(), 
        "cv_std": cv.std()
        }


# Train different models with top features
def test_models(X_train, X_test, y_train, y_test, features, models):
    results = []

    for k in range(1, len(features)+1):
        top_k = features[:k]
        X_tr_sub = X_train[top_k]
        X_te_sub = X_test[top_k]

        print(f"\nTop {k} features: {top_k}")

        for name, model in models.items():
            metrics = evaluate(
                model, X_tr_sub, X_te_sub, y_train, y_test
                )
            
            print(f"{name}: Test Acc={metrics['accuracy']:.3f}")
            
            results.append({
                "model": name,
                "num_features": k,
                "features": ", ".join(top_k),
                "test_acc": metrics["accuracy"],
                "test_auc": metrics["auc"],
                "cv_mean": metrics["cv_mean"],
                "cv_std": metrics["cv_std"]
            })

    return pd.DataFrame(results)


# Save the best model and related info
def save_best(results_df, X_train, y_train, models, file="best_model.pkl"):
    best = results_df.loc[results_df["test_acc"].idxmax()]
    
    print("\nBest model configuration:")
    print(best)
    
    best_model = models[best["model"]]
    final_features = best["features"].split(", ")
    best_model.fit(X_train[final_features], y_train)
    
    joblib.dump(best_model, file)
    
    # Save feature importance
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        imp = abs(best_model.coef_[0])
    else:
        imp = [1/len(final_features)]*len(final_features)
    
    final_importance = pd.DataFrame({"feature": final_features, "importance": imp})
    final_importance.to_csv("feature_importance.csv", index=False)
    
    results_df.to_csv("model_results.csv", index=False)
    
    print("Saved model, results, and feature importance.")


# Main workflow
def main():
    file_path = "ACME-HappinessSurvey2020.csv"
    df, X_train, X_test, y_train, y_test = load_data(file_path)
    
    features = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']
    plot_counts(df, features)
    plot_corr(df)
    feature_vs_target(df, features)
    
    top_features = get_feature_ranking(X_train, y_train)
    
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, 
            class_weight="balanced",
            random_state=42
            ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, 
            class_weight="balanced",
            random_state=42
            ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
            ),
        "XGBoost": XGBClassifier(
            n_estimators=100, 
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]))
    }
    
    results = test_models(X_train, X_test, y_train, y_test, top_features, models)

    save_best(results, X_train, y_train, models)


if __name__ == "__main__":
    main()