import cv2
import numpy as np

def morphological_noise_suppression_adaptive(img, window_size=3, noise_threshold=15.0):
    """
    Адаптивное морфологическое подавление шума.
    noise_threshold: Порог дисперсии. Если дисперсия ниже порога, пиксель не меняется.
    """
    img_f = img.astype(np.float32)
    h, w = img.shape
    
    # Вычисляем среднее и дисперсию для центрированного окна
    mean_H = cv2.blur(img_f, (window_size, window_size))
    mean_sq_H = cv2.blur(img_f**2, (window_size, window_size))
    var_H = mean_sq_H - mean_H**2 # Центральная дисперсия
    
    pad = window_size // 2
    min_var = np.full((h, w), np.inf, dtype=np.float32)
    result_img = np.zeros((h, w), dtype=np.float32)
    
    # Классический поиск сдвига с минимальной дисперсией (тяжелый алгоритм)
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            y_start, y_end = max(0, -dy), min(h, h - dy)
            x_start, x_end = max(0, -dx), min(w, w - dx)
            
            sy_start, sy_end = max(0, dy), min(h, h + dy)
            sx_start, sx_end = max(0, dx), min(w, w + dx)
            
            shifted_var = np.full((h, w), np.inf, dtype=np.float32)
            shifted_mean = np.zeros((h, w), dtype=np.float32)
            
            shifted_var[y_start:y_end, x_start:x_end] = var_H[sy_start:sy_end, sx_start:sx_end]
            shifted_mean[y_start:y_end, x_start:x_end] = mean_H[sy_start:sy_end, sx_start:sx_end]
            
            mask = shifted_var < min_var
            min_var[mask] = shifted_var[mask]
            result_img[mask] = shifted_mean[mask]
            
    # --- НОВОЕ: АДАПТИВНОЕ ПРИМЕНЕНИЕ ---
    # Создаем маску "чистых" областей, где изначальная дисперсия была меньше порога
    clean_mask = var_H < noise_threshold
    
    # Если область "чистая", оставляем оригинальный пиксель. 
    # Если "шумная" - берем результат морфологического фильтра.
    final_result = np.where(clean_mask, img_f, result_img)
    
    return np.clip(final_result, 0, 255).astype(np.uint8)

def compute_mse(img1, img2):
    """Вычисление среднеквадратичного отклонения (MSE)."""
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64))**2)

# ==========================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ И СРАВНЕНИЯ МЕТОДОВ
# ==========================================
if __name__ == "__main__":
    # Замените 'original.png' и 'noisy.png' на пути к вашим файлам!
    # Если идеального изображения нет, скрипт ниже покажет, как сгенерировать шум.
    
    # Чтение изображений (в оттенках серого)
    img_true = cv2.imread('text-a-true.png', cv2.IMREAD_GRAYSCALE)
    img_noisy = cv2.imread('text-a-sp.png', cv2.IMREAD_GRAYSCALE)
    
    if img_true is not None and img_noisy is not None:
        window_size = 3
        
        print("Начинаем обработку...")
        # 1. Задание 10: Морфологическое подавление шума
        iterations = 3
        img_result = img_noisy.copy()

        for i in range(iterations):
            img_result = morphological_noise_suppression_adaptive(img_result, window_size=3)
        
        # 2. Задание 3: Линейный фильтр (например, однородное усреднение)
        img_linear = cv2.blur(img_noisy, (window_size, 5))
        
        # 3. Задание 6: Медианный фильтр
        img_median = cv2.medianBlur(img_noisy, 5)
        
        # Вывод MSE
        # print(f"MSE до обработки (сам шум): {compute_mse(img_true, img_noisy):.2f}")
        # print(f"MSE линейного фильтра (Задание 3): {compute_mse(img_true, img_linear):.2f}")
        # print(f"MSE медианного фильтра (Задание 6): {compute_mse(img_true, img_median):.2f}")
        # print(f"MSE морфологического фильтра: {compute_mse(img_true, img_morph):.2f}")
        
        # Сохранение результатов
        cv2.imwrite('result_morph.png', img_result)
        cv2.imwrite('result_linear.png', img_linear)
        cv2.imwrite('result_median.png', img_median)
        print("Готово. Результаты сохранены в текущую папку.")
    else:
        print("Ошибка: Не удалось найти изображения 'original.png' и 'noisy.png'. Пожалуйста, проверьте пути к файлам.")
        