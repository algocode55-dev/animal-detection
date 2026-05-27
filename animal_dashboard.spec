# animal_dashboard.spec  — PyInstaller build spec
# Build: pyinstaller animal_dashboard.spec

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['dashboard.py'],
    pathex=[str(Path('.').resolve())],
    binaries=[],
    datas=[
        ('assets',  'assets'),
        ('models',  'models'),
        ('utils',   'utils'),
    ],
    hiddenimports=[
        'ultralytics',
        'cv2',
        'numpy',
        'torch',
        'torchvision',
        'PIL',
        'simpleaudio',
        'PyQt6.QtMultimedia',
        'PyQt6.sip',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'scipy'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='AnimalDetectionDashboard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # No console window on Windows
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='assets/icons/app.ico',   # Uncomment when icon is available
)
