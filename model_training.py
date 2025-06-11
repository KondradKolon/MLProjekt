from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier

# Usuń importy TensorFlow:
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """Trenuje różne modele i zwraca wyniki"""
    # Skalowanie danych
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Słownik z modelami
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Neural Network (MLP)': MLPClassifier(hidden_layer_sizes=(100, 50), 
                                         max_iter=1000, 
                                         activation='relu', 
                                         solver='adam',
                                         random_state=42),
        'Deep Neural Network': MLPClassifier(
                                        hidden_layer_sizes=(200, 100, 50), 
                                        max_iter=2000,
                                        activation='relu', 
                                        solver='adam',
                                        alpha=0.0001,
                                        learning_rate='adaptive',
                                        early_stopping=True,
                                        validation_fraction=0.1,
                                        random_state=42)                                
        
    }  # Naprawiłem też nawias klamrowy
    
    # Słownik z wynikami
    results = {}
    predictions = {}
    
    # Trenowanie i ocena każdego modelu
    for name, model in models.items():
        print(f"Trenowanie modelu {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        predictions[name] = y_pred
    
    return results, predictions, X_train_scaled, X_test_scaled