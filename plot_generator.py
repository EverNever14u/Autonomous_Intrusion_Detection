import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_plots(X, X_scaled, y, le, feature_idx=0):
    '''
    generate plot
    '''
    plt.figure(figsize=(10, 5)) 
    label_names = le.inverse_transform(y)
    
    sns.countplot(
        y=label_names, 
        order=pd.Series(label_names).value_counts().index, 
        palette='viridis',
        hue=label_names,
        legend=False
    )
    plt.title('Attacks vs Normal Traffic')
    plt.xlabel('Number of samples')
    plt.ylabel('Traffic type')
    plt.tight_layout()

    feature_name = X.columns[feature_idx]

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(X.iloc[:, feature_idx], bins=50, ax=axes2[0], color='orange', kde=True)
    axes2[0].set_title(f'Before scaling: {feature_name}')
    axes2[0].set_ylabel('Count')

    sns.histplot(X_scaled[:, feature_idx], bins=50, ax=axes2[1], color='blue', kde=True)
    axes2[1].set_title(f'After scaling: {feature_name}')
    axes2[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()