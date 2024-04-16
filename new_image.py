import streamlit as st
from skimage import io
import numpy as np

def compress_image(image, top_k):
    U, sing_values, V = np.linalg.svd(image)
    sigma = np.zeros_like(image, dtype=float)
    np.fill_diagonal(sigma, sing_values)
    # Усеченное SVD
    truncated_U = U[:, :top_k]
    truncated_sigma = sigma[:top_k, :top_k]
    truncated_V = V[:top_k, :]
    # Восстановление сжатого изображения
    compressed_image = truncated_U @ truncated_sigma @ truncated_V
    return np.clip(compressed_image, 0, 255).astype(np.uint8)

# Заголовок и инструкция
st.title('SVD Compression for Images')
st.write("Загрузите изображение по URL и выберите количество сингулярных чисел для сжатия.")

# Виджет для ввода URL изображения
url = st.text_input("Введите URL изображения:")


# Кнопка для загрузки изображения
if url:
    try:
        # Получение изображения по URL
        image = io.imread(url)[:, :, 0]
        max_top_k = image.shape[0]
        top_k = st.slider('Количество сингулярных чисел', 1, max_top_k, 50)
        # Преобразование изображения в массив numpy
        # image = np.array(img)
        # Сжатие изображения
        compressed_image = compress_image(image, top_k)
        # Отображение сжатого изображения
        st.image(compressed_image, caption=f'Сжатое изображение с {top_k} сингулярными числами', use_column_width=True)
        st.image(image, caption=f'Исходное изображение', use_column_width=True)
    except Exception as e:
        st.error(f"Произошла ошибка при загрузке или обработке изображения: {e}")

