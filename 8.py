import cv2
import numpy as np
from skimage import color
import time

def compute_delta_for_window(f_win, g_win):
    """
    Вычисляет сумму квадратов r для каналов L, a, b в заданном окне.
    delta = r_L^2 + r_a^2 + r_b^2
    """
    delta = 0.0
    # Проходим по каналам: 0-L, 1-a, 2-b
    for c in range(3):
        F = f_win[:, :, c].flatten()
        G = g_win[:, :, c].flatten()
        
        # Вычисляем все попарные разности F_i - F_j и G_i - G_j
        diff_F = F[:, np.newaxis] - F[np.newaxis, :]
        diff_G = G[:, np.newaxis] - G[np.newaxis, :]
        
        # Ищем нарушения монотонности: F_i > F_j, но G_i < G_j
        # В матричном виде: diff_F > 0 и diff_G < 0
        mask = (diff_F > 0) & (diff_G < 0)
        
        if np.any(mask):
            # Размер нарушения r_c = max(min(F_i - F_j, G_j - G_i))
            r_c = np.max(np.minimum(diff_F[mask], -diff_G[mask]))
        else:
            r_c = 0.0
            
        delta += r_c ** 2
        
    return delta

def process_video(video_path, window_size=5, threshold=300, step=2):
    """
    Основная функция обработки видео.
    :param video_path: Путь к видеофайлу
    :param window_size: Размер скользящего окна (W x W)
    :param threshold: Порог для суммы квадратов r (дельта)
    :param step: Шаг смещения окна (увеличьте для ускорения, 1 - для макс. точности)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видеофайл.")
        return

    # Читаем первый кадр для выбора ROI и инициализации фона
    ret, bg_frame = cap.read()
    if not ret:
        print("Ошибка: Видео пустое.")
        return

    print("Инструкция:")
    print("1. Выделите область интереса (ROI) левой кнопкой мыши.")
    print("2. Нажмите SPACE или ENTER для подтверждения.")
    print("3. Нажмите 'q' во время воспроизведения для выхода.")

    # Выбор области
    roi = cv2.selectROI("Select ROI", bg_frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")
    x_roi, y_roi, w_roi, h_roi = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
    
    if w_roi == 0 or h_roi == 0:
        print("Область не выбрана. Выход.")
        return

    # Вырезаем фон и переводим в CIELAB
    bg_roi = bg_frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
    # Для skimage значения RGB должны быть [0, 1] для корректного перевода
    bg_lab = color.rgb2lab(bg_roi[..., ::-1] / 255.0) 

    half_w = window_size // 2

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Видео закончилось.")
            break
            
        start_time = time.time()

        # Вырезаем текущий ROI и переводим в CIELAB
        curr_roi = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        curr_lab = color.rgb2lab(curr_roi[..., ::-1] / 255.0)
        
        # Карта отличий
        diff_map = np.zeros((h_roi, w_roi), dtype=np.uint8)

        # Скользящее окно с заданным шагом
        for y in range(half_w, h_roi - half_w, step):
            for x in range(half_w, w_roi - half_w, step):
                # Извлекаем окна
                bg_win = bg_lab[y-half_w:y+half_w+1, x-half_w:x+half_w+1]
                curr_win = curr_lab[y-half_w:y+half_w+1, x-half_w:x+half_w+1]
                
                # Считаем дельту
                delta = compute_delta_for_window(bg_win, curr_win)
                
                # Если отличие превышает порог, помечаем область
                if delta > threshold:
                    # Закрашиваем блок размером step x step для визуализации
                    diff_map[y:y+step, x:x+step] = 255

        # Отрисовка результатов
        # Создаем красную маску для выделения изменений
        red_mask = np.zeros_like(curr_roi)
        red_mask[:, :, 2] = diff_map # BGR формат, 2 - красный канал
        
        # Накладываем маску на текущий ROI
        blended_roi = cv2.addWeighted(curr_roi, 1.0, red_mask, 0.5, 0)
        
        # Вставляем обработанный ROI обратно в кадр
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (x_roi, y_roi), (x_roi+w_roi, y_roi+h_roi), (0, 255, 0), 2)
        display_frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi] = blended_roi

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Pytyev Morphological Analysis (Var XX)", display_frame)
        cv2.imshow("Difference Mask", diff_map)

        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- ЗАПУСК ---
if __name__ == "__main__":
    # Укажите путь к вашему видео (можно использовать веб-камеру, указав 0)
    VIDEO_FILE = "8.flv" 
    
    # Рекомендации по параметрам:
    # window_size: 3, 5, 7. Чем больше, тем надежнее, но медленнее.
    # threshold: Зависит от шума камеры. Попробуйте значения от 100 до 1000.
    # step: 2-4 для приемлемого FPS.
    process_video(VIDEO_FILE, window_size=5, threshold=150, step=3)