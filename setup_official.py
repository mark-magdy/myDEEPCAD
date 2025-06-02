import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    requirements = [
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'tensorboard>=2.5.0',
        'tqdm>=4.62.0',
        'numpy>=1.19.5',
        'matplotlib>=3.4.3',
        'scikit-learn>=0.24.2'
    ]
    
    print("Installing required packages...")
    for package in requirements:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("All packages installed successfully!")

def create_directory_structure():
    """Create necessary directories for the project"""
    directories = [
        'data/deepcad/train',
        'data/deepcad/val',
        'checkpoints',
        'logs',
        'models'
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Directory structure created successfully!")

def main():
    print("Setting up DeepCAD project...")
    
    # Install requirements
    install_requirements()
    
    # Create directory structure
    create_directory_structure()
    
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Place your training data in data/deepcad/train/")
    print("2. Place your validation data in data/deepcad/val/")
    print("3. Run training with: python train.py")

if __name__ == "__main__":
    main()
