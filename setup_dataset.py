"""
setup_dataset.py
================
Automated dataset setup for the Gym Training CNN.

Creates all dataset folders and downloads gym equipment images
from Unsplash (free, no login required, no 6GB download needed).

Classes:
  barbell, bench, dumbbell, kettlebell,
  pull_up_bar, rowing_machine, squat_rack, treadmill

Run ONCE before train_cnn.py:
    python setup_dataset.py
"""

import os
import shutil
import urllib.request

TRAIN_DIR    = 'dataset/train'
TEST_DIR     = 'dataset/test'
TEST_IMG_DIR = 'test_images'

CLASS_NAMES = [
    'barbell', 'bench', 'dumbbell', 'kettlebell',
    'pull_up_bar', 'rowing_machine', 'squat_rack', 'treadmill'
]

TRAIN_COUNT = 120
TEST_COUNT  = 25

# Unsplash image URLs per class (free, no account needed)
# Each URL loads a resized version (~100-200 KB per image)
IMAGE_URLS = {
    'dumbbell': [
        'https://images.unsplash.com/photo-1517836357463-d25dfeac3438?w=300',
        'https://images.unsplash.com/photo-1534438327276-14e5300c3a48?w=300',
        'https://images.unsplash.com/photo-1583454110551-21f2fa2afe61?w=300',
        'https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=300',
        'https://images.unsplash.com/photo-1578762560042-46ad127c95ea?w=300',
        'https://images.unsplash.com/photo-1590239926044-4131fe192b73?w=300',
    ],
    'treadmill': [
        'https://images.unsplash.com/photo-1540497077202-7c8a3999166f?w=300',
        'https://images.unsplash.com/photo-1561214115-f2f134cc4912?w=300',
        'https://images.unsplash.com/photo-1518611012118-696072aa579a?w=300',
        'https://images.unsplash.com/photo-1576678927484-cc907957088c?w=300',
        'https://images.unsplash.com/photo-1571019614242-c5c5dee9f50b?w=300',
        'https://images.unsplash.com/photo-1595078475328-1ab05d0a6a0e?w=300',
    ],
    'barbell': [
        'https://images.unsplash.com/photo-1526506118085-60ce8714f8c5?w=300',
        'https://images.unsplash.com/photo-1574680096145-d05b474e2155?w=300',
        'https://images.unsplash.com/photo-1581009137042-c552e485697a?w=300',
        'https://images.unsplash.com/photo-1599058917212-d750089bc07e?w=300',
        'https://images.unsplash.com/photo-1567598508481-65985588e295?w=300',
        'https://images.unsplash.com/photo-1593079831268-3381b0db4a77?w=300',
    ],
    'bench': [
        'https://images.unsplash.com/photo-1534438327276-14e5300c3a48?w=300',
        'https://images.unsplash.com/photo-1571902943202-507ec2618e8f?w=300',
        'https://images.unsplash.com/photo-1605296867304-46d5465a13f1?w=300',
        'https://images.unsplash.com/photo-1597452485677-d661670d9640?w=300',
        'https://images.unsplash.com/photo-1577221084712-45b0445d2b00?w=300',
        'https://images.unsplash.com/photo-1534367610401-9f5ed68180aa?w=300',
    ],
    'kettlebell': [
        'https://images.unsplash.com/photo-1521804906057-1df8fdb718b7?w=300',
        'https://images.unsplash.com/photo-1614926857083-7be149266cda?w=300',
        'https://images.unsplash.com/photo-1584735935682-2f2b69dff9d2?w=300',
        'https://images.unsplash.com/photo-1598289431512-b97b0917afea?w=300',
        'https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=300',
        'https://images.unsplash.com/photo-1540496905036-5937c10647cc?w=300',
    ],
    'pull_up_bar': [
        'https://images.unsplash.com/photo-1598971639058-fab3c3109a00?w=300',
        'https://images.unsplash.com/photo-1541534741688-6078c6bfb5c5?w=300',
        'https://images.unsplash.com/photo-1566241142559-40e1dab266c6?w=300',
        'https://images.unsplash.com/photo-1595078475328-1ab05d0a6a0e?w=300',
        'https://images.unsplash.com/photo-1605296867304-46d5465a13f1?w=300',
        'https://images.unsplash.com/photo-1576678927484-cc907957088c?w=300',
    ],
    'rowing_machine': [
        'https://images.unsplash.com/photo-1593079831268-3381b0db4a77?w=300',
        'https://images.unsplash.com/photo-1540497077202-7c8a3999166f?w=300',
        'https://images.unsplash.com/photo-1571019614242-c5c5dee9f50b?w=300',
        'https://images.unsplash.com/photo-1518611012118-696072aa579a?w=300',
        'https://images.unsplash.com/photo-1595078475328-1ab05d0a6a0e?w=300',
        'https://images.unsplash.com/photo-1561214115-f2f134cc4912?w=300',
    ],
    'squat_rack': [
        'https://images.unsplash.com/photo-1526506118085-60ce8714f8c5?w=300',
        'https://images.unsplash.com/photo-1581009137042-c552e485697a?w=300',
        'https://images.unsplash.com/photo-1567598508481-65985588e295?w=300',
        'https://images.unsplash.com/photo-1574680096145-d05b474e2155?w=300',
        'https://images.unsplash.com/photo-1599058917212-d750089bc07e?w=300',
        'https://images.unsplash.com/photo-1534367610401-9f5ed68180aa?w=300',
    ],
}


def create_folders():
    """Create all dataset and test_images directories."""
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR,  cls), exist_ok=True)
    os.makedirs(TEST_IMG_DIR, exist_ok=True)
    print("[setup] Folder structure created.")


def download_image(url: str, save_path: str) -> bool:
    """Download a single image from a URL."""
    try:
        req = urllib.request.Request(
            url, headers={'User-Agent': 'Mozilla/5.0 GymBotSetup/1.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            with open(save_path, 'wb') as f:
                f.write(resp.read())
        return True
    except Exception as e:
        print(f"  [!] Failed: {url[:60]}... ({e})")
        return False


def populate_class(class_name: str, urls: list):
    """Download images for one class, split into train/test."""
    train_dir = os.path.join(TRAIN_DIR, class_name)
    test_dir  = os.path.join(TEST_DIR,  class_name)
    total_needed = TRAIN_COUNT + TEST_COUNT
    # Repeat URL list to reach needed count
    url_list = (urls * (total_needed // len(urls) + 2))[:total_needed]

    print(f"\n[setup] Downloading '{class_name}' ({total_needed} images)...")
    for i, url in enumerate(url_list):
        filename = f"{class_name}_{i+1:04d}.jpg"
        dest = os.path.join(train_dir if i < TRAIN_COUNT else test_dir, filename)
        if os.path.exists(dest):
            continue
        download_image(url, dest)
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1} done")

    t = len(os.listdir(train_dir))
    v = len(os.listdir(test_dir))
    print(f"  [OK] {class_name}: {t} train, {v} test")


def copy_test_samples():
    """Copy one image per class into test_images/ for demo."""
    print("\n[setup] Copying sample images to test_images/...")
    for cls in CLASS_NAMES:
        src_dir = os.path.join(TRAIN_DIR, cls)
        imgs = [f for f in os.listdir(src_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if imgs:
            shutil.copy2(os.path.join(src_dir, imgs[0]),
                         os.path.join(TEST_IMG_DIR, f"{cls}.jpg"))
            print(f"  Copied {cls}.jpg")


def main():
    print("=" * 55)
    print("  Dataset Setup — Gym Training CNN")
    print("=" * 55)
    create_folders()
    for cls in CLASS_NAMES:
        populate_class(cls, IMAGE_URLS.get(cls, []))
    copy_test_samples()

    print("\n" + "=" * 55)
    print("  Setup complete!")
    print("=" * 55)
    print(f"\n{'Class':20s} {'Train':>6} {'Test':>6}")
    print("-" * 35)
    for cls in CLASS_NAMES:
        t = len(os.listdir(os.path.join(TRAIN_DIR, cls)))
        v = len(os.listdir(os.path.join(TEST_DIR, cls)))
        print(f"{cls:20s} {t:>6} {v:>6}")
    print("\nNext step: python train_cnn.py")


if __name__ == '__main__':
    main()
