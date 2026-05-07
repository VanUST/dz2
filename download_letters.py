import os
import random
import numpy as np
from PIL import Image
import torch
import torchvision

def download_and_save_letters(letter1, letter2, n=10, output_dir="letters"):
    letter1 = letter1.upper().strip()
    letter2 = letter2.upper().strip()
    
    if not (letter1.isalpha() and letter2.isalpha()) or len(letter1) != 1 or len(letter2) != 1:
        raise ValueError("Введите ровно по одной английской букве (A-Z)")
        
    if not (8 <= n <= 12):
        print(f"⚠️  Рекомендовано 8-12 образцов. Будет использовано: {max(8, min(n, 12))}")
        n = max(8, min(n, 12))

    # Создаем директории
    os.makedirs(output_dir, exist_ok=True)
    dir1 = os.path.join(output_dir, f"letter_{letter1}")
    dir2 = os.path.join(output_dir, f"letter_{letter2}")
    os.makedirs(dir1, exist_ok=True)
    os.makedirs(dir2, exist_ok=True)

    print("📦 Загрузка EMNIST (split='letters') через torchvision...")
    # split='letters' содержит только A-Z (26 классов)
    dataset = torchvision.datasets.EMNIST(
        root='./emnist_data', 
        split='letters', 
        train=True, 
        download=True, 
        transform=None  # Получаем сырые тензоры для удобной обработки
    )

    print(f"🔍 Поиск буквы '{letter1}'...")
    # В EMNIST letters метки идут 1-26 для A-Z
    target1 = ord(letter1) - ord('A') + 1
    indices1 = torch.where(dataset.targets == target1)[0].tolist()
    
    print(f"🔍 Поиск буквы '{letter2}'...")
    target2 = ord(letter2) - ord('A') + 1
    indices2 = torch.where(dataset.targets == target2)[0].tolist()

    if len(indices1) < n or len(indices2) < n:
        print(f"⚠️  Доступно меньше примеров, чем запрошено. Будет взято всё доступное.")
        n = min(n, len(indices1), len(indices2))

    # Случайная выборка
    random.seed(42)
    selected1 = random.sample(indices1, n)
    selected2 = random.sample(indices2, n)

    print("💾 Сохранение изображений...")
    # dataset.data имеет форму (N, 28, 28), значения 0-255
    data_np = dataset.data.numpy()

    for i, idx in enumerate(selected1):
        img = Image.fromarray(data_np[idx].astype(np.uint8))
        img.save(os.path.join(dir1, f"{letter1.lower()}_{i+1:02d}.png"))

    for i, idx in enumerate(selected2):
        img = Image.fromarray(data_np[idx].astype(np.uint8))
        img.save(os.path.join(dir2, f"{letter2.lower()}_{i+1:02d}.png"))

    print(f"\n✅ Готово! Сохранено:")
    print(f"   📂 {dir1}/  ({n} файлов)")
    print(f"   📂 {dir2}/  ({n} файлов)")
    print("💡 Изображения уже имеют одинаковый масштаб (28x28) и центрированы.")

if __name__ == "__main__":
    print("="*50)
    print("📝 Загрузчик рукописных букв (EMNIST)")
    print("="*50)
    
    l1 = input("Первая буква (A-Z): ").strip()
    l2 = input("Вторая буква (A-Z): ").strip()
    try:
        n = int(input("Количество образцов на букву (8-12): ").strip())
    except ValueError:
        n = 10
        print("Используем по умолчанию: 10")
        
    out_dir = input("Папка вывода (Enter = 'letters'): ").strip() or "letters"
    
    download_and_save_letters(l1, l2, n, out_dir)