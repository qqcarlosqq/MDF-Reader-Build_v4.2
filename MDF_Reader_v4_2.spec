# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
datas=[]; binaries=[]; hiddenimports=[]
for package in ("asammdf","numpy","pandas","canmatrix","lxml"):
    d,b,h=collect_all(package); datas+=d; binaries+=b; hiddenimports+=h
a=Analysis(["src/MDF_Reader_v4_2.py"],pathex=[],binaries=binaries,datas=datas,hiddenimports=hiddenimports,hookspath=[],hooksconfig={},runtime_hooks=[],excludes=["PyQt5","PyQt6","PySide2","PySide6","matplotlib"],noarchive=False,optimize=1)
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,a.binaries,a.datas,[],name="MDF_Reader_v4_2",debug=False,bootloader_ignore_signals=False,strip=False,upx=False,console=True)
