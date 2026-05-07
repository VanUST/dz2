import cv2
import numpy as np

def compute_block_morph_difference(f_block, g_block, k=3):
    pixels_f = f_block.reshape((-1, 3)).astype(np.float32)
    pixels_g = g_block.reshape((-1, 3)).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    if np.var(pixels_f) < 1e-5:
        labels = np.zeros(pixels_f.shape[0], dtype=np.int32)
    else:
        _, labels, _ = cv2.kmeans(pixels_f, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
    
    labels = labels.flatten()
    morph_diff = 0.0
    
    for cluster_id in range(k):
        mask = (labels == cluster_id)
        if not np.any(mask):
            continue
            
        g_cluster_pixels = pixels_g[mask]
        mean_g = np.mean(g_cluster_pixels, axis=0)
        morph_diff += np.sum((g_cluster_pixels - mean_g) ** 2)
        
    # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
    # Делим на общее количество значений (пиксели * каналы), чтобы получить MSE
    mse = morph_diff / (pixels_f.shape[0] * 3)
    return mse

def process_video_with_roi(video_path, block_size=16, k=2, threshold=50): # Порог теперь сильно меньше!
    cap = cv2.VideoCapture(video_path)
    ret, bg_frame = cap.read()
    if not ret:
        return

    # Размываем фон по Гауссу, чтобы убить высокочастотный микрошум камеры
    bg_frame = cv2.GaussianBlur(bg_frame, (5, 5), 0)

    roi = cv2.selectROI("Select ROI", bg_frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")
    
    x_roi, y_roi, w_roi, h_roi = [int(v) for v in roi]
    w_adj = w_roi - (w_roi % block_size)
    h_adj = h_roi - (h_roi % block_size)
    
    bg_roi = bg_frame[y_roi:y_roi+h_adj, x_roi:x_roi+w_adj]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Также размываем текущий кадр от шума
        frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        curr_roi = frame_blurred[y_roi:y_roi+h_adj, x_roi:x_roi+w_adj]
        
        diff_map = np.zeros((h_adj // block_size, w_adj // block_size), dtype=np.float32)
        
        for y in range(0, h_adj, block_size):
            for x in range(0, w_adj, block_size):
                f_blk = bg_roi[y:y+block_size, x:x+block_size]
                g_blk = curr_roi[y:y+block_size, x:x+block_size]
                
                diff = compute_block_morph_difference(f_blk, g_blk, k)
                diff_map[y//block_size, x//block_size] = diff
                
        # --- ВИЗУАЛИЗАЦИЯ 1: Тепловая карта (Без порога) ---
        # Нормализуем массив отличий от 0 до 255, чтобы увидеть реальную картину
        heatmap = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_resized = cv2.resize(heatmap, (w_adj, h_adj), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Raw Heatmap (See the actual diff)', heatmap_resized)

        # --- ВИЗУАЛИЗАЦИЯ 2: Пороговая (Красные квадраты) ---
        mask = (diff_map > threshold).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask, (w_adj, h_adj), interpolation=cv2.INTER_NEAREST)
        
        result_roi = frame[y_roi:y_roi+h_adj, x_roi:x_roi+w_adj].copy() # Рисуем поверх резкого кадра
        result_roi[mask_resized == 255] = [0, 0, 255]
        
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (x_roi, y_roi), (x_roi+w_adj, y_roi+h_adj), (0, 255, 0), 2)
        display_frame[y_roi:y_roi+h_adj, x_roi:x_roi+w_adj] = result_roi

        cv2.imshow('Result', display_frame)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Запуск
process_video_with_roi("8.flv", block_size=16, k=2, threshold=50)