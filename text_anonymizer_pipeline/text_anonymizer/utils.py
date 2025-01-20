import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path, test_size=0.2, random_state=42):
    """Загружает датасет и делит его на тренировочные и тестовые данные."""
    try:
        data = pd.read_csv(path)
        train, test = train_test_split(data, test_size=test_size, random_state=random_state)
        return train, test
    except Exception as e:
        print(f"Ошибка при загрузке датасета: {e}")
        raise

def load_input_text(path):
    """Загружает текстовые данные из файла."""
    try:
        with open(path, 'r', encoding='utf-8') as file:
            texts = file.readlines()
        return [text.strip() for text in texts]
    except Exception as e:
        print(f"Ошибка при загрузке текстов: {e}")
        raise

def save_output(texts, path):
    """Сохраняет анонимизированные тексты в файл."""
    try:
        with open(path, 'w', encoding='utf-8') as file:
            for text in texts:
                file.write(text + '\n')
        print(f"Анонимизированные тексты успешно сохранены в {path}")
    except Exception as e:
        print(f"Ошибка при сохранении текстов: {e}")
        raise
