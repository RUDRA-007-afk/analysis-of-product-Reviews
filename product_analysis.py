# Product Review Sentiment Analysis
# ===================================

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Natural Language Processing libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Download required NLTK data
print("📦 Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Visualization setup
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🎯 Product Review Sentiment Analysis")
print("=" * 50)

# 1. DATA GENERATION (Simulated E-commerce Reviews)
# =================================================
print("\n1. Generating Sample Product Reviews...")

# Create realistic product review data
np.random.seed(42)

# Sample products
products = [
    "Wireless Bluetooth Headphones",
    "Smartphone Case",
    "Laptop Stand",
    "Wireless Mouse",
    "USB-C Cable",
    "Portable Charger",
    "Bluetooth Speaker",
    "Webcam HD",
    "Mechanical Keyboard",
    "Monitor Stand"
]

# Sample positive review templates
positive_reviews = [
    "Excellent product! Highly recommend it to everyone.",
    "Amazing quality and fast delivery. Very satisfied!",
    "Perfect! Exactly what I was looking for.",
    "Great value for money. Works perfectly.",
    "Outstanding quality and performance. Love it!",
    "Fantastic product! Exceeded my expectations.",
    "Really good quality. Very happy with this purchase.",
    "Superb! Great design and functionality.",
    "Wonderful product. Fast shipping and great service.",
    "Incredible quality for the price. Highly recommended!",
    "Best purchase I've made this year!",
    "Amazing! Works exactly as described.",
    "Top-notch quality. Very impressed.",
    "Excellent build quality and performance.",
    "Perfect fit and finish. Love the design!"
]

# Sample negative review templates
negative_reviews = [
    "Terrible quality. Broke after one day.",
    "Worst purchase ever. Complete waste of money.",
    "Poor quality and doesn't work as advertised.",
    "Horrible! Stopped working after a week.",
    "Very disappointed. Not worth the money.",
    "Awful product. Would not recommend.",
    "Complete garbage. Save your money.",
    "Defective product. Poor customer service.",
    "Cheap quality. Fell apart immediately.",
    "Useless product. Doesn't work at all.",
    "Very poor quality control.",
    "Disappointed with the performance.",
    "Not as described. Poor build quality.",
    "Broke on first use. Terrible!",
    "Waste of money. Very unhappy."
]

# Sample neutral review templates
neutral_reviews = [
    "It's okay. Nothing special but does the job.",
    "Average product. Could be better.",
    "Decent quality for the price.",
    "It works fine. Standard product.",
    "Not bad, but not great either.",
    "Acceptable quality. Gets the job done.",
    "It's alright. Would consider other options next time.",
    "Fair product. Some pros and cons.",
    "Reasonable quality. Expected more features.",
    "Good enough for basic use.",
    "Standard product. Nothing to complain about.",
    "Works as expected. No surprises.",
    "Mediocre quality. Room for improvement.",
    "Functional but could be better designed.",
    "Basic product that meets minimum requirements."
]

# Generate review dataset
n_reviews = 1000
reviews_data = []

for i in range(n_reviews):
    # Randomly select sentiment distribution (60% positive, 25% negative, 15% neutral)
    sentiment_choice = np.random.choice(['positive', 'negative', 'neutral'], 
                                       p=[0.6, 0.25, 0.15])
    
    # Select review text based on sentiment
    if sentiment_choice == 'positive':
        review_text = np.random.choice(positive_reviews)
        rating = np.random.choice([4, 5], p=[0.3, 0.7])
    elif sentiment_choice == 'negative':
        review_text = np.random.choice(negative_reviews)
        rating = np.random.choice([1, 2], p=[0.6, 0.4])
    else:  # neutral
        review_text = np.random.choice(neutral_reviews)
        rating = 3
    
    # Add some variation to reviews
    variations = [
        " The delivery was fast.",
        " Packaging was good.",
        " Price is reasonable.",
        " Customer service was helpful.",
        " Would buy again.",
        " Shipping took longer than expected.",
        " Easy to use.",
        " Good instructions included.",
        " Sturdy construction.",
        " Lightweight design."
    ]
    
    # Sometimes add variation
    if np.random.random() < 0.3:
        review_text += np.random.choice(variations)
    
    reviews_data.append({
        'product_name': np.random.choice(products),
        'review_text': review_text,
        'rating': rating,
        'reviewer_name': f"Customer_{i+1}",
        'review_date': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 365)),
        'actual_sentiment': sentiment_choice  # Ground truth for validation
    })

# Create DataFrame
df = pd.DataFrame(reviews_data)

print(f"✅ Generated {len(df)} product reviews")
print(f"📊 Actual sentiment distribution:")
print(df['actual_sentiment'].value_counts())

# Display sample reviews
print(f"\n📝 Sample Reviews:")
for i in range(3):
    print(f"\nReview {i+1}:")
    print(f"Product: {df.iloc[i]['product_name']}")
    print(f"Rating: {df.iloc[i]['rating']}/5")
    print(f"Review: {df.iloc[i]['review_text']}")
    print(f"Actual Sentiment: {df.iloc[i]['actual_sentiment']}")

# 2. DATA PREPROCESSING
# =====================
print("\n\n2. Data Preprocessing...")

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_column):
        """Preprocess entire dataframe"""
        df['cleaned_review'] = df[text_column].apply(self.clean_text)
        return df

# Initialize preprocessor and clean data
preprocessor = TextPreprocessor()
df = preprocessor.preprocess_dataframe(df, 'review_text')

print("✅ Text preprocessing completed")
print(f"\nOriginal text example: '{df.iloc[0]['review_text']}'")
print(f"Cleaned text example: '{df.iloc[0]['cleaned_review']}'")

# 3. SENTIMENT ANALYSIS USING MULTIPLE METHODS
# =============================================
print("\n\n3. Performing Sentiment Analysis...")

# Method 1: TextBlob Sentiment Analysis
def analyze_sentiment_textblob(text):
    """Analyze sentiment using TextBlob"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

# Method 2: VADER Sentiment Analysis
sia = SentimentIntensityAnalyzer()

def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER"""
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Method 3: Rating-based sentiment (as baseline)
def rating_to_sentiment(rating):
    """Convert rating to sentiment"""
    if rating >= 4:
        return 'positive'
    elif rating <= 2:
        return 'negative'
    else:
        return 'neutral'

# Apply all sentiment analysis methods
print("Applying TextBlob sentiment analysis...")
df['textblob_sentiment'] = df['review_text'].apply(analyze_sentiment_textblob)

print("Applying VADER sentiment analysis...")
df['vader_sentiment'] = df['review_text'].apply(analyze_sentiment_vader)

print("Converting ratings to sentiment...")
df['rating_sentiment'] = df['rating'].apply(rating_to_sentiment)

print("✅ Sentiment analysis completed using multiple methods")

# 4. SENTIMENT ANALYSIS RESULTS & COMPARISON
# ===========================================
print("\n\n4. Sentiment Analysis Results...")

# Compare different methods
methods = ['actual_sentiment', 'textblob_sentiment', 'vader_sentiment', 'rating_sentiment']

print("📊 Sentiment Distribution Comparison:")
comparison_data = {}
for method in methods:
    sentiment_counts = df[method].value_counts()
    comparison_data[method] = sentiment_counts
    print(f"\n{method.replace('_', ' ').title()}:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")

# Create comparison DataFrame
comparison_df = pd.DataFrame(comparison_data).fillna(0)
print(f"\n📋 Complete Comparison Table:")
print(comparison_df)

# 5. MODEL ACCURACY EVALUATION
# =============================
print("\n\n5. Model Accuracy Evaluation...")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_sentiment_model(actual, predicted, model_name):
    """Evaluate sentiment analysis model"""
    accuracy = accuracy_score(actual, predicted)
    print(f"\n📈 {model_name} Performance:")
    print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    print(f"\nClassification Report:")
    print(classification_report(actual, predicted))
    
    return accuracy

# Evaluate each method against actual sentiment
textblob_accuracy = evaluate_sentiment_model(
    df['actual_sentiment'], df['textblob_sentiment'], "TextBlob"
)

vader_accuracy = evaluate_sentiment_model(
    df['actual_sentiment'], df['vader_sentiment'], "VADER"
)

rating_accuracy = evaluate_sentiment_model(
    df['actual_sentiment'], df['rating_sentiment'], "Rating-based"
)

# Find best method
accuracies = {
    'TextBlob': textblob_accuracy,
    'VADER': vader_accuracy,
    'Rating-based': rating_accuracy
}

best_method = max(accuracies, key=accuracies.get)
print(f"\n🏆 Best Method: {best_method} (Accuracy: {accuracies[best_method]:.3f})")

# 6. COMPREHENSIVE VISUALIZATIONS
# ===============================
print("\n\n6. Creating Visualizations...")

# Set up the plotting area
fig = plt.figure(figsize=(20, 15))

# 1. Sentiment Distribution Comparison
plt.subplot(3, 3, 1)
comparison_df.plot(kind='bar', ax=plt.gca(), color=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
plt.title('Sentiment Distribution Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)

# 2. Pie chart for actual sentiment distribution
plt.subplot(3, 3, 2)
actual_sentiment_counts = df['actual_sentiment'].value_counts()
colors = ['lightblue', 'lightcoral', 'lightgreen']
plt.pie(actual_sentiment_counts.values, labels=actual_sentiment_counts.index, 
        autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Actual Sentiment Distribution', fontsize=14, fontweight='bold')

# 3. Rating distribution
plt.subplot(3, 3, 3)
rating_counts = df['rating'].value_counts().sort_index()
bars = plt.bar(rating_counts.index, rating_counts.values, 
               color=['red', 'orange', 'yellow', 'lightgreen', 'green'])
plt.title('Rating Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Rating')
plt.ylabel('Count')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{int(height)}', ha='center', va='bottom')

# 4. Sentiment by Product
plt.subplot(3, 3, 4)
product_sentiment = pd.crosstab(df['product_name'], df['actual_sentiment'])
product_sentiment_pct = product_sentiment.div(product_sentiment.sum(axis=1), axis=0)
product_sentiment_pct.plot(kind='bar', stacked=True, ax=plt.gca(), 
                          color=['lightblue', 'lightcoral', 'lightgreen'])
plt.title('Sentiment Distribution by Product', fontsize=14, fontweight='bold')
plt.xlabel('Product')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')

# 5. Model Accuracy Comparison
plt.subplot(3, 3, 5)
methods_list = list(accuracies.keys())
accuracy_values = list(accuracies.values())
bars = plt.bar(methods_list, accuracy_values, color=['skyblue', 'lightcoral', 'lightgreen'])
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Method')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom')

# 6. Confusion Matrix for Best Method
plt.subplot(3, 3, 6)
best_method_col = f"{best_method.lower().replace('-', '_')}_sentiment"
cm = confusion_matrix(df['actual_sentiment'], df[best_method_col])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title(f'Confusion Matrix - {best_method}', fontsize=14, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# 7. Word frequency in positive reviews
plt.subplot(3, 3, 7)
positive_reviews_text = ' '.join(df[df['actual_sentiment'] == 'positive']['cleaned_review'])
positive_words = positive_reviews_text.split()
positive_word_freq = Counter(positive_words).most_common(10)
words, counts = zip(*positive_word_freq)
plt.barh(words, counts, color='lightgreen')
plt.title('Top Words in Positive Reviews', fontsize=14, fontweight='bold')
plt.xlabel('Frequency')

# 8. Word frequency in negative reviews
plt.subplot(3, 3, 8)
negative_reviews_text = ' '.join(df[df['actual_sentiment'] == 'negative']['cleaned_review'])
negative_words = negative_reviews_text.split()
negative_word_freq = Counter(negative_words).most_common(10)
words, counts = zip(*negative_word_freq)
plt.barh(words, counts, color='lightcoral')
plt.title('Top Words in Negative Reviews', fontsize=14, fontweight='bold')
plt.xlabel('Frequency')

# 9. Sentiment over time (by month)
plt.subplot(3, 3, 9)
df['review_month'] = df['review_date'].dt.to_period('M')
monthly_sentiment = df.groupby(['review_month', 'actual_sentiment']).size().unstack(fill_value=0)
monthly_sentiment.plot(kind='line', ax=plt.gca(), marker='o',
                      color=['lightblue', 'lightcoral', 'lightgreen'])
plt.title('Sentiment Trends Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Number of Reviews')
plt.legend(title='Sentiment')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 7. DETAILED ANALYSIS & INSIGHTS
# ================================
print("\n\n7. Detailed Analysis & Insights...")

# Sentiment by rating analysis
print("📊 Sentiment Analysis by Rating:")
rating_sentiment_analysis = pd.crosstab(df['rating'], df['actual_sentiment'], normalize='index')
print(rating_sentiment_analysis.round(3))

# Product performance analysis
print(f"\n🏆 Product Performance Analysis:")
product_stats = df.groupby('product_name').agg({
    'rating': 'mean',
    'actual_sentiment': lambda x: (x == 'positive').sum() / len(x)
}).round(3)
product_stats.columns = ['Avg_Rating', 'Positive_Sentiment_Rate']
product_stats = product_stats.sort_values('Positive_Sentiment_Rate', ascending=False)
print(product_stats)

# Word analysis for each sentiment
print(f"\n📝 Key Words Analysis:")
for sentiment in ['positive', 'negative', 'neutral']:
    sentiment_text = ' '.join(df[df['actual_sentiment'] == sentiment]['cleaned_review'])
    words = sentiment_text.split()
    top_words = Counter(words).most_common(5)
    print(f"\nTop words in {sentiment} reviews:")
    for word, count in top_words:
        print(f"  - {word}: {count}")

# 8. SAVE RESULTS
# ===============
print("\n\n8. Saving Results...")

# Save processed dataset
df.to_csv('sentiment_analysis_results.csv', index=False)
print("✅ Results saved to 'sentiment_analysis_results.csv'")

# Save model performance summary
performance_summary = pd.DataFrame({
    'Method': list(accuracies.keys()),
    'Accuracy': list(accuracies.values())
})
performance_summary.to_csv('model_performance.csv', index=False)
print("✅ Model performance saved to 'model_performance.csv'")

# Save product analysis
product_stats.to_csv('product_performance_analysis.csv')
print("✅ Product analysis saved to 'product_performance_analysis.csv'")

# 9. EXAMPLE USAGE FOR NEW REVIEWS
# =================================
print("\n\n9. Example: Analyzing New Reviews...")

def analyze_new_review(review_text, method='vader'):
    """Analyze sentiment of a new review"""
    if method.lower() == 'textblob':
        sentiment = analyze_sentiment_textblob(review_text)
    elif method.lower() == 'vader':
        sentiment = analyze_sentiment_vader(review_text)
    else:
        sentiment = "Unknown method"
    
    return sentiment

# Example new reviews
new_reviews = [
    "This product is absolutely amazing! Best purchase ever!",
    "Terrible quality. Broke immediately. Very disappointed.",
    "It's okay. Does what it's supposed to do, nothing more."
]

print(f"🔍 Analyzing New Reviews using {best_method}:")
for i, review in enumerate(new_reviews, 1):
    method_name = best_method.lower().replace('-based', '').replace(' ', '')
    sentiment = analyze_new_review(review, method_name)
    print(f"\nReview {i}: '{review}'")
    print(f"Predicted Sentiment: {sentiment.capitalize()}")

# 10. SUMMARY REPORT
# ==================
print("\n\n" + "="*60)
print("📋 SENTIMENT ANALYSIS SUMMARY REPORT")
print("="*60)

print(f"\n📊 Dataset Overview:")
print(f"  • Total Reviews Analyzed: {len(df):,}")
print(f"  • Products Covered: {df['product_name'].nunique()}")
print(f"  • Date Range: {df['review_date'].min().date()} to {df['review_date'].max().date()}")

print(f"\n🎯 Sentiment Distribution:")
for sentiment, count in df['actual_sentiment'].value_counts().items():
    percentage = (count / len(df)) * 100
    print(f"  • {sentiment.capitalize()}: {count:,} reviews ({percentage:.1f}%)")

print(f"\n🏆 Best Performing Model:")
print(f"  • Method: {best_method}")
print(f"  • Accuracy: {accuracies[best_method]:.1%}")

print(f"\n📈 Key Insights:")
top_product = product_stats.index[0]
worst_product = product_stats.index[-1]
print(f"  • Best Rated Product: {top_product}")
print(f"    - Positive Sentiment Rate: {product_stats.loc[top_product, 'Positive_Sentiment_Rate']:.1%}")
print(f"  • Needs Improvement: {worst_product}")
print(f"    - Positive Sentiment Rate: {product_stats.loc[worst_product, 'Positive_Sentiment_Rate']:.1%}")

avg_positive_rate = df['actual_sentiment'].value_counts(normalize=True)['positive']
print(f"  • Overall Customer Satisfaction: {avg_positive_rate:.1%}")

print(f"\n💡 Recommendations:")
print(f"  • Focus on improving products with low positive sentiment rates")
print(f"  • Leverage insights from highly rated products")
print(f"  • Monitor sentiment trends over time for early warning signs")
print(f"  • Use {best_method} for ongoing sentiment monitoring")

print(f"\n📁 Output Files Generated:")
print(f"  • sentiment_analysis_results.csv - Complete analysis results")
print(f"  • model_performance.csv - Model comparison metrics")
print(f"  • product_performance_analysis.csv - Product-wise performance")

print(f"\n🎉 Analysis Complete!")
print("="*60)