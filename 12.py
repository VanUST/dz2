import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# --- 1. ЗАГРУЗКА ДАННЫХ ---
def load_emnist_letters(base_path, letter_char):
    folder = os.path.join(base_path, f"letter_{letter_char}")
    images = []
    if not os.path.exists(folder):
        print(f"Ошибка: Папка {folder} не найдена!")
        return np.array([])
        
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32) / 255.0
            images.append(img.flatten())
            
    return np.array(images)

# --- 2. ПОСТРОЕНИЕ ПРОЕКТОРОВ С РЕГУЛЯРИЗАЦИЕЙ ---
def build_morph_projector(train_images):
    n_pixels = train_images.shape[1]
    vectors = [np.ones(n_pixels)] # Фон
    
    for img in train_images:
        mask = (img > 0.2).astype(np.float32)
        vectors.append(mask)
        
    A = np.column_stack(vectors)
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    
    # --- МАГИЯ ЗДЕСЬ: Регуляризация (Усечение спектра) ---
    rank = 4
    
    return U[:, :rank], S

def get_morph_ratio(g, Q, E_vec):
    P_g = Q @ (np.dot(Q.T, g))
    E_g = E_vec * np.dot(E_vec, g)
    
    numerator = np.sum((g - P_g)**2)
    denominator = np.sum((P_g - E_g)**2)
    
    return numerator / (denominator + 1e-9)

# --- ЭКСПЕРИМЕНТ ---
def run_experiment():
    n = 12
    m = 9 
    base_path = "letters"
    img_size = 28
    n_pixels = img_size * img_size
    
    data_I = load_emnist_letters(base_path, "I")
    data_J = load_emnist_letters(base_path, "J")
    
    if len(data_I) < n or len(data_J) < n:
        print("Недостаточно данных!")
        return

    train_I, test_I = data_I[:m], data_I[m:]
    train_J, test_J = data_J[:m], data_J[m:]

    Q_I, S_I = build_morph_projector(train_I)
    Q_J, S_J = build_morph_projector(train_J)
    E_vec = np.ones(n_pixels) / np.sqrt(n_pixels)

    X_train = np.vstack([train_I, train_J])
    y_train = np.array([0]*m + [1]*m)
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X_train, y_train)
    w = clf.coef_[0]

    # --- УВЕЛИЧИВАЕМ ДИАПАЗОН ШУМА ---
    sigmas = np.linspace(0.1, 3.5, 15) 
    repeats = 200 # Больше повторений для гладких графиков
    results = {'morph_I': [], 'morph_J': [], 'lin_I': [], 'lin_J': []}

    print("Запуск симуляции (это может занять около минуты)...")
    
    for sigma in sigmas:
        err_m_I, err_m_J, err_l_I, err_l_J = 0, 0, 0, 0
        for i in range(n - m):
            for _ in range(repeats):
                # Для буквы I
                g_I = test_I[i] + np.random.normal(0, sigma, n_pixels)
                if get_morph_ratio(g_I, Q_J, E_vec) < get_morph_ratio(g_I, Q_I, E_vec): err_m_I += 1
                if clf.predict([g_I])[0] != 0: err_l_I += 1
                
                # Для буквы J
                g_J = test_J[i] + np.random.normal(0, sigma, n_pixels)
                if get_morph_ratio(g_J, Q_I, E_vec) < get_morph_ratio(g_J, Q_J, E_vec): err_m_J += 1
                if clf.predict([g_J])[0] != 1: err_l_J += 1
        
        total = (n - m) * repeats
        results['morph_I'].append(err_m_I / total)
        results['morph_J'].append(err_m_J / total)
        results['lin_I'].append(err_l_I / total)
        results['lin_J'].append(err_l_J / total)

    # --- ВИЗУАЛИЗАЦИЯ ---
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(w.reshape(img_size, img_size), cmap='RdBu')
    plt.title("Разделяющее изображение w (SVM)")
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.plot(S_I / S_I[0], 'ro-', label="Буква I (До усечения)")
    plt.plot(S_J / S_J[0], 'bo-', label="Буква J (До усечения)")
    plt.axvline(x=Q_I.shape[1]-1, color='k', linestyle='--', label="Граница регуляризации")
    plt.title("Сингулярные числа (Эффективная размерность)")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(sigmas, results['morph_I'], 'r-', linewidth=2, label="Морфологический I")
    plt.plot(sigmas, results['lin_I'], 'r--', linewidth=2, label="Линейный I")
    plt.title("Ошибки классификации: Буква I")
    plt.xlabel("Уровень шума (Sigma)"); plt.ylabel("Доля ошибок")
    plt.grid(True); plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(sigmas, results['morph_J'], 'b-', linewidth=2, label="Морфологический J")
    plt.plot(sigmas, results['lin_J'], 'b--', linewidth=2, label="Линейный J")
    plt.title("Ошибки классификации: Буква J")
    plt.xlabel("Уровень шума (Sigma)"); plt.ylabel("Доля ошибок")
    plt.grid(True); plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()