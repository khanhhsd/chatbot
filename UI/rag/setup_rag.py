"""
Setup script to install RAG dependencies.
Run this before using RAG functionality.
"""

import subprocess
import sys
import platform


def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✓ {description} completed")
            return True
        else:
            print(f"✗ {description} failed")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ {description} timed out")
        return False
    except Exception as e:
        print(f"✗ {description} error: {e}")
        return False


def setup_minimal():
    """Install minimal RAG dependencies."""
    print("\n" + "="*60)
    print("Setting up Minimal RAG (in-memory, CPU-only)")
    print("="*60)
    
    packages = [
        "sentence-transformers",
        "scikit-learn",
        "pandas",
    ]
    
    for package in packages:
        cmd = f"{sys.executable} -m pip install {package}"
        run_command(cmd, f"Installing {package}")


def setup_production():
    """Install production RAG with FAISS."""
    print("\n" + "="*60)
    print("Setting up Production RAG (FAISS acceleration)")
    print("="*60)
    
    setup_minimal()
    
    # FAISS installation depends on system
    print("\nInstalling FAISS...")
    if platform.system() == "Windows":
        # Windows typically uses CPU version
        cmd = f"{sys.executable} -m pip install faiss-cpu"
    else:
        # Linux can use either
        cmd = f"{sys.executable} -m pip install faiss-cpu"
    
    run_command(cmd, "Installing FAISS (CPU version)")


def setup_gpu():
    """Install GPU-accelerated FAISS."""
    print("\n" + "="*60)
    print("Setting up GPU-Accelerated RAG")
    print("="*60)
    
    setup_minimal()
    
    print("\nInstalling FAISS (GPU)...")
    print("Note: This requires CUDA toolkit installed")
    print("See: https://developer.nvidia.com/cuda-downloads")
    
    cmd = f"{sys.executable} -m pip install faiss-gpu"
    run_command(cmd, "Installing FAISS GPU")


def verify_installation():
    """Verify that all RAG dependencies are installed."""
    print("\n" + "="*60)
    print("Verifying Installation")
    print("="*60)
    
    packages = {
        "sentence_transformers": "Sentence Transformers (embeddings)",
        "sklearn": "Scikit-learn (similarity)",
        "pandas": "Pandas (data processing)",
    }
    
    all_ok = True
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name}")
            all_ok = False
    
    # Optional: FAISS
    try:
        import faiss
        print(f"✓ FAISS (optional but recommended for production)")
    except ImportError:
        print(f"○ FAISS (optional - not installed)")
    
    return all_ok


def main():
    """Main setup flow."""
    print("\n" + "#"*60)
    print("# RAG Setup Script")
    print("#"*60)
    
    print("\nSelect setup option:")
    print("1. Minimal RAG (in-memory, faster setup)")
    print("2. Production RAG (FAISS, optimized)")
    print("3. GPU-Accelerated RAG (FAISS + GPU)")
    print("4. Just verify installation")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        setup_minimal()
    elif choice == "2":
        setup_production()
    elif choice == "3":
        setup_gpu()
    elif choice == "4":
        verify_installation()
    elif choice == "5":
        print("\nExiting...")
        return
    else:
        print("\nInvalid choice")
        return
    
    # Verify
    print("\nVerifying installation...")
    if verify_installation():
        print("\n✓ Setup completed successfully!")
        print("\nYou can now:")
        print("- Run 'python test_rag_quick_start.py' to test RAG")
        print("- Enable RAG in config.py: USE_RAG = True")
        print("- Run the full application: python main.py")
    else:
        print("\n✗ Some packages failed to install")
        print("Try installing them manually:")
        print("  pip install sentence-transformers scikit-learn pandas")


if __name__ == "__main__":
    main()
