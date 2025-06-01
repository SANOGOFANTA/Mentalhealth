import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Configuration des styles
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class MentalHealthEDA:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self):
        """Charger les données"""
        print("📊 Chargement des données...")
        self.df = pd.read_csv(self.data_path)
        self.df.columns = self.df.columns.str.strip()
        print(f"✓ Dataset chargé: {self.df.shape[0]} lignes, {self.df.shape[1]} colonnes")
        return self.df
    
    def basic_info(self):
        """Afficher les informations de base"""
        print("\n📋 INFORMATIONS DE BASE")
        print("="*50)
        print(f"Nombre total d'observations: {len(self.df)}")
        print(f"Nombre de colonnes: {len(self.df.columns)}")
        print(f"\nColonnes: {self.df.columns.tolist()}")
        print(f"\nTypes de données:\n{self.df.dtypes}")
        print(f"\nValeurs manquantes:\n{self.df.isnull().sum()}")
        
    def plot_class_distribution(self):
        """Visualiser la distribution des classes"""
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Bar plot
        plt.subplot(1, 2, 1)
        class_counts = self.df['status'].value_counts()
        colors = sns.color_palette("viridis", len(class_counts))
        bars = plt.bar(class_counts.index, class_counts.values, color=colors)
        
        # Ajouter les valeurs sur les barres
        for bar, count in zip(bars, class_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                    f'{count}\n({count/len(self.df)*100:.1f}%)', 
                    ha='center', va='bottom', fontsize=10)
        
        plt.title('Distribution des Classes de Santé Mentale', fontsize=14, fontweight='bold')
        plt.xlabel('Status', fontsize=12)
        plt.ylabel('Nombre d\'observations', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Subplot 2: Pie chart
        plt.subplot(1, 2, 2)
        plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('Répartition en Pourcentage', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('Results/class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Afficher le déséquilibre
        print("\n⚖️ ANALYSE DU DÉSÉQUILIBRE DES CLASSES")
        print("="*50)
        max_class = class_counts.max()
        for status, count in class_counts.items():
            ratio = count / max_class
            print(f"{status}: {count} ({count/len(self.df)*100:.1f}%) - Ratio: {ratio:.2f}")
    
    def analyze_text_length(self):
        """Analyser la longueur des textes"""
        self.df['text_length'] = self.df['statement'].str.len()
        self.df['word_count'] = self.df['statement'].str.split().str.len()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribution de la longueur des textes
        axes[0, 0].hist(self.df['text_length'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Distribution de la Longueur des Textes (caractères)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Longueur (caractères)')
        axes[0, 0].set_ylabel('Fréquence')
        axes[0, 0].axvline(self.df['text_length'].mean(), color='red', linestyle='--', 
                          label=f'Moyenne: {self.df["text_length"].mean():.0f}')
        axes[0, 0].legend()
        
        # Distribution du nombre de mots
        axes[0, 1].hist(self.df['word_count'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Distribution du Nombre de Mots', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Nombre de mots')
        axes[0, 1].set_ylabel('Fréquence')
        axes[0, 1].axvline(self.df['word_count'].mean(), color='red', linestyle='--', 
                          label=f'Moyenne: {self.df["word_count"].mean():.0f}')
        axes[0, 1].legend()
        
        # Boxplot par classe
        axes[1, 0].set_title('Longueur des Textes par Classe', fontsize=12, fontweight='bold')
        self.df.boxplot(column='text_length', by='status', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Status')
        axes[1, 0].set_ylabel('Longueur (caractères)')
        plt.sca(axes[1, 0])
        plt.xticks(rotation=45, ha='right')
        
        # Violin plot pour le nombre de mots
        axes[1, 1].set_title('Distribution du Nombre de Mots par Classe', fontsize=12, fontweight='bold')
        sns.violinplot(data=self.df, x='status', y='word_count', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Status')
        axes[1, 1].set_ylabel('Nombre de mots')
        plt.sca(axes[1, 1])
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('Results/text_length_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Statistiques
        print("\n📏 STATISTIQUES DE LONGUEUR DE TEXTE")
        print("="*50)
        print(f"Longueur moyenne (caractères): {self.df['text_length'].mean():.0f}")
        print(f"Longueur médiane (caractères): {self.df['text_length'].median():.0f}")
        print(f"Nombre moyen de mots: {self.df['word_count'].mean():.0f}")
        print(f"Nombre médian de mots: {self.df['word_count'].median():.0f}")
        
    def create_wordclouds(self):
        """Créer des nuages de mots pour chaque classe"""
        unique_classes = self.df['status'].unique()
        n_classes = len(unique_classes)
        
        # Calculer le nombre de lignes et colonnes pour la grille
        n_cols = 3
        n_rows = (n_classes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_classes > 1 else [axes]
        
        for idx, status in enumerate(unique_classes):
            # Filtrer les textes pour cette classe
            texts = ' '.join(self.df[self.df['status'] == status]['statement'].tolist())
            
            # Nettoyer le texte
            texts = re.sub(r'[^a-zA-Z\s]', '', texts.lower())
            
            # Créer le nuage de mots
            wordcloud = WordCloud(width=400, height=300, background_color='white',
                                 stopwords=self.stop_words, max_words=100,
                                 colormap='viridis').generate(texts)
            
            axes[idx].imshow(wordcloud, interpolation='bilinear')
            axes[idx].set_title(f'Nuage de Mots - {status}', fontsize=12, fontweight='bold')
            axes[idx].axis('off')
        
        # Masquer les axes supplémentaires
        for idx in range(n_classes, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('Results/wordclouds_by_class.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def analyze_common_words(self):
        """Analyser les mots les plus fréquents"""
        from collections import Counter
        
        # Fonction pour obtenir les mots les plus fréquents
        def get_top_words(texts, n=20):
            all_words = []
            for text in texts:
                if pd.notna(text):
                    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
                    words = [w for w in words if w not in self.stop_words and len(w) > 2]
                    all_words.extend(words)
            return Counter(all_words).most_common(n)
        
        # Mots les plus fréquents globalement
        all_texts = self.df['statement'].tolist()
        top_words_global = get_top_words(all_texts)
        
        # Créer le graphique
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Graphique 1: Top mots globaux
        words, counts = zip(*top_words_global)
        axes[0].barh(words, counts, color='lightcoral')
        axes[0].set_title('20 Mots les Plus Fréquents (Global)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Fréquence')
        axes[0].invert_yaxis()
        
        # Graphique 2: Top mots par classe (top 3 classes)
        top_classes = self.df['status'].value_counts().head(3).index
        x_pos = np.arange(10)
        width = 0.25
        
        for i, status in enumerate(top_classes):
            class_texts = self.df[self.df['status'] == status]['statement'].tolist()
            top_words_class = get_top_words(class_texts, 10)
            words, counts = zip(*top_words_class)
            axes[1].bar(x_pos + i*width, counts, width, label=status)
        
        axes[1].set_title('10 Mots les Plus Fréquents par Classe', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Mots')
        axes[1].set_ylabel('Fréquence')
        axes[1].set_xticks(x_pos + width)
        axes[1].set_xticklabels([get_top_words(all_texts, 10)[i][0] for i in range(10)], rotation=45, ha='right')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('Results/common_words_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def analyze_sentiment_indicators(self):
        """Analyser les indicateurs de sentiment"""
        # Mots clés par catégorie
        anxiety_words = ['anxious', 'worried', 'nervous', 'panic', 'fear', 'scared', 'restless']
        depression_words = ['sad', 'depressed', 'hopeless', 'empty', 'worthless', 'lonely']
        positive_words = ['happy', 'grateful', 'good', 'great', 'wonderful', 'beautiful', 'love']
        
        # Compter les occurrences
        for word_list, name in [(anxiety_words, 'anxiety'), 
                                (depression_words, 'depression'), 
                                (positive_words, 'positive')]:
            self.df[f'{name}_score'] = self.df['statement'].apply(
                lambda x: sum(1 for word in word_list if word in str(x).lower())
            )
        
        # Créer le graphique
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Heatmap des scores moyens par classe
        score_cols = ['anxiety_score', 'depression_score', 'positive_score']
        heatmap_data = self.df.groupby('status')[score_cols].mean()
        
        sns.heatmap(heatmap_data.T, annot=True, fmt='.2f', cmap='YlOrRd', 
                   ax=axes[0, 0], cbar_kws={'label': 'Score moyen'})
        axes[0, 0].set_title('Scores Moyens d\'Indicateurs par Classe', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Status')
        axes[0, 0].set_ylabel('Type d\'indicateur')
        
        # Distribution des scores
        for idx, (score_col, ax) in enumerate(zip(score_cols, [axes[0, 1], axes[1, 0], axes[1, 1]])):
            for status in self.df['status'].unique()[:3]:  # Top 3 classes
                data = self.df[self.df['status'] == status][score_col]
                ax.hist(data, alpha=0.5, label=status, bins=10)
            
            ax.set_title(f'Distribution des {score_col.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Score')
            ax.set_ylabel('Fréquence')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('Results/sentiment_indicators.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_correlation_matrix(self):
        """Créer une matrice de corrélation"""
        # Préparer les données numériques
        numeric_data = self.df[['text_length', 'word_count', 
                               'anxiety_score', 'depression_score', 'positive_score']].copy()
        
        # Ajouter l'encodage des classes
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        numeric_data['status_encoded'] = le.fit_transform(self.df['status'])
        
        # Calculer la corrélation
        correlation_matrix = numeric_data.corr()
        
        # Créer le graphique
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, linewidths=1,
                   cbar_kws={"shrink": .8})
        plt.title('Matrice de Corrélation des Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('Results/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_summary_report(self):
        """Créer un rapport de synthèse"""
        plt.figure(figsize=(12, 10))
        
        # Titre principal
        plt.suptitle('Rapport d\'Analyse Exploratoire - Mental Health Dataset', 
                    fontsize=16, fontweight='bold')
        
        # Subplot 1: Résumé statistique
        ax1 = plt.subplot(3, 2, 1)
        ax1.axis('off')
        summary_text = f"""
        📊 RÉSUMÉ DU DATASET
        
        • Nombre total d'observations: {len(self.df)}
        • Nombre de classes: {self.df['status'].nunique()}
        • Longueur moyenne des textes: {self.df['text_length'].mean():.0f} caractères
        • Nombre moyen de mots: {self.df['word_count'].mean():.0f}
        • Classe majoritaire: {self.df['status'].value_counts().index[0]}
        • Ratio de déséquilibre: 1:{self.df['status'].value_counts().max() / self.df['status'].value_counts().min():.1f}
        """
        ax1.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
        
        # Subplot 2: Distribution des classes (mini)
        ax2 = plt.subplot(3, 2, 2)
        class_counts = self.df['status'].value_counts().head(5)
        ax2.barh(class_counts.index, class_counts.values, color='lightblue')
        ax2.set_title('Top 5 Classes', fontsize=10)
        ax2.set_xlabel('Nombre')
        
        # Subplot 3: Distribution des longueurs
        ax3 = plt.subplot(3, 2, 3)
        ax3.hist(self.df['text_length'], bins=30, color='lightgreen', alpha=0.7)
        ax3.set_title('Distribution des Longueurs', fontsize=10)
        ax3.set_xlabel('Longueur (caractères)')
        ax3.set_ylabel('Fréquence')
        
        # Subplot 4: Boxplot des scores
        ax4 = plt.subplot(3, 2, 4)
        score_data = self.df[['anxiety_score', 'depression_score', 'positive_score']].values
        ax4.boxplot(score_data, labels=['Anxiété', 'Dépression', 'Positif'])
        ax4.set_title('Distribution des Scores d\'Indicateurs', fontsize=10)
        ax4.set_ylabel('Score')
        
        # Subplot 5: Insights clés
        ax5 = plt.subplot(3, 2, 5)
        ax5.axis('off')
        insights_text = """
        💡 INSIGHTS CLÉS
        
        • Dataset fortement déséquilibré
        • Variation importante dans la longueur des textes
        • Présence de mots-clés émotionnels distincts
        • Nécessité d'équilibrage pour l'entraînement
        • Preprocessing recommandé pour normaliser
        """
        ax5.text(0.1, 0.5, insights_text, fontsize=10, verticalalignment='center')
        
        # Subplot 6: Recommandations
        ax6 = plt.subplot(3, 2, 6)
        ax6.axis('off')
        recommendations_text = """
        🎯 RECOMMANDATIONS
        
        • Utiliser des techniques d'équilibrage (SMOTE, class weights)
        • Appliquer TF-IDF avec n-grams
        • Considérer des modèles robustes au déséquilibre
        • Validation stratifiée obligatoire
        • Monitoring des métriques par classe
        """
        ax6.text(0.1, 0.5, recommendations_text, fontsize=10, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('Results/eda_summary_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def run_complete_eda(self):
        """Exécuter l'analyse complète"""
        import os
        os.makedirs('Results', exist_ok=True)
        
        print("🔍 DÉBUT DE L'ANALYSE EXPLORATOIRE DES DONNÉES")
        print("="*60)
        
        # Charger les données
        self.load_data()
        
        # Informations de base
        self.basic_info()
        
        # Analyses et visualisations
        print("\n📊 Création des visualisations...")
        self.plot_class_distribution()
        self.analyze_text_length()
        self.create_wordclouds()
        self.analyze_common_words()
        self.analyze_sentiment_indicators()
        self.create_correlation_matrix()
        self.create_summary_report()
        
        print("\n✅ ANALYSE COMPLÈTE TERMINÉE!")
        print(f"📁 Les visualisations sont sauvegardées dans: Results/")
        
        return self.df

# Exécution
if __name__ == "__main__":
    eda = MentalHealthEDA('Data/Mentalhealth.csv')
    df = eda.run_complete_eda()