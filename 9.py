import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from PIL import Image, ImageDraw, ImageFont

# --- 1. ГЕНЕРАЦИЯ И КЭШИРОВАНИЕ БАЗОВЫХ ИЗОБРАЖЕНИЙ БУКВ ---
def generate_letters(n_letters=20, size=32, cache_dir="letters_cache"):
    letters = "ABCDEFGHIJKLMNOPRSTU"
    masks = []
    
    # Создаем директорию, если её нет
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Создана директория для кэша: {cache_dir}")
    
    # Пытаемся загрузить шрифт
    try:
        font = ImageFont.truetype("arial.ttf", size - 4)
    except IOError:
        font = ImageFont.load_default()

    for i in range(min(n_letters, len(letters))):
        text = letters[i]
        file_path = os.path.join(cache_dir, f"{text}.png")
        
        # Если файл существует, загружаем его из кэша
        if os.path.exists(file_path):
            img = Image.open(file_path).convert('L')
            print(f"Загружено из кэша: {file_path}")
        else:
            # Если нет, генерируем новое изображение
            img = Image.new('L', (size, size), 0)
            draw = ImageDraw.Draw(img)
            
            # Центрируем текст
            bbox = draw.textbbox((0, 0), text, font=font)
            x = (size - (bbox[2] - bbox[0])) / 2
            y = (size - (bbox[3] - bbox[1])) / 2
            draw.text((x, y), text, fill=255, font=font)
            
            # Сохраняем в кэш
            img.save(file_path)
            print(f"Сгенерировано и сохранено: {file_path}")
            
        # Бинаризуем изображение для получения маски
        mask = np.array(img) > 127
        masks.append(mask.astype(float))
        
    return np.array(masks)

# --- 2. ПОСТРОЕНИЕ БАЗИСОВ ФОРМ L_i ---
def build_bases(masks):
    n_letters, h, w = masks.shape
    bases = []
    
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    for mask in masks:
        chi_F = mask.flatten()
        chi_B = 1.0 - chi_F
        
        v1 = chi_F
        v2 = chi_B
        v3 = chi_B * X_flat
        v4 = chi_B * Y_flat
        
        A = np.column_stack([v1, v2, v3, v4])
        Q, R = np.linalg.qr(A)
        bases.append(Q)
        
    return bases

# --- 3. ГЕНЕРАЦИЯ ТЕСТОВОЙ ВЫБОРКИ ---
def generate_test_image(mask, sigma):
    h, w = mask.shape
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    
    chi_F = mask
    chi_B = 1.0 - chi_F
    
    f_F = np.random.uniform(0.5, 1.0)
    q1 = np.random.uniform(0.1, 0.4)
    dx = np.random.uniform(-0.2, 0.2)
    dy = np.random.uniform(-0.2, 0.2)
    
    clean_img = chi_F * f_F + chi_B * (q1 + dx * X + dy * Y)
    noise = np.random.normal(0, sigma, clean_img.shape)
    
    return clean_img + noise

# --- 4. ВЫЧИСЛЕНИЕ ИНДЕКСОВ МОРФОЛОГИЧЕСКОЙ НЕЗАВИСИМОСТИ ---
def compute_morph_independence(Q_i, Q_j):
    S = np.linalg.svd(Q_i.T @ Q_j, compute_uv=False)
    cosines = [s for s in S if s < 0.999]
    
    if not cosines:
        return 0.0
        
    max_cos = max(cosines)
    independence_index = 1.0 - max_cos**2
    return independence_index

# --- ОСНОВНОЙ ЭКСПЕРИМЕНТ ---
def main():
    n_letters = 20
    n_samples_per_class = 100
    
    # Загружаем или генерируем маски из директории letters_cache
    masks = generate_letters(n_letters, cache_dir="letters_cache")
    bases = build_bases(masks)
    
    sigmas = np.linspace(0.1, 1.0, 15)
    error_rates = []
    
    print("\nЗапуск симуляции классификации. Пожалуйста, подождите...")
    
    max_sigma_confusion = np.zeros((n_letters, n_letters))
    target_sigma_errors = np.zeros((n_letters, n_letters))
    target_sigma_found = False
    
    for sigma in sigmas:
        errors = 0
        conf_matrix = np.zeros((n_letters, n_letters))
        
        for true_label in range(n_letters):
            for _ in range(n_samples_per_class):
                img = generate_test_image(masks[true_label], sigma)
                img_flat = img.flatten()
                
                distances = []
                for Q in bases:
                    proj = Q @ (Q.T @ img_flat)
                    dist = np.sum((img_flat - proj)**2)
                    distances.append(dist)
                    
                pred_label = np.argmin(distances)
                conf_matrix[true_label, pred_label] += 1
                
                if pred_label != true_label:
                    errors += 1
                    
        err_rate = errors / (n_letters * n_samples_per_class)
        error_rates.append(err_rate)
        
        if err_rate >= 0.15 and not target_sigma_found:
            target_sigma_errors = conf_matrix.copy()
            target_sigma_found = True
            
        if sigma == sigmas[-1]:
            max_sigma_confusion = conf_matrix.copy()

    # Корреляция Спирмена
    indep_indices = []
    pairwise_errors = []
    
    for i in range(n_letters):
        for j in range(n_letters):
            if i != j:
                idx = compute_morph_independence(bases[i], bases[j])
                indep_indices.append(idx)
                err_ij = target_sigma_errors[i, j] + target_sigma_errors[j, i]
                pairwise_errors.append(err_ij)
                
    spearman_corr, p_value = spearmanr(indep_indices, pairwise_errors)
    
    # Визуализация
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(sigmas, error_rates, marker='o', color='b')
    plt.title('Частота ошибок от уровня шума $\sigma$')
    plt.xlabel('$\sigma$ (Гауссовский шум)')
    plt.ylabel('Доля ошибок')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.imshow(max_sigma_confusion, cmap='Blues', interpolation='nearest')
    plt.title(f'Матрица путаницы (max $\sigma$={sigmas[-1]:.2f})')
    plt.colorbar()
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    
    plt.subplot(1, 3, 3)
    plt.scatter(indep_indices, pairwise_errors, color='r')
    plt.title(f'Спирмен: r={spearman_corr:.2f}, p={p_value:.3f}')
    plt.xlabel('Индекс морф. независимости')
    plt.ylabel('Число взаимных ошибок')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()