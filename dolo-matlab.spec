# -*- mode: python -*-
a = Analysis(['bin/dolo-matlab'],
             hiddenimports=['scipy.special._ufuncs_cxx'],
             hookspath=None,
             runtime_hooks=None)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='dolo-matlab.exe',
          debug=False,
          strip=None,
          upx=True,
          console=True )
	  
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
	       [('recipes.yaml', 'dolo\\compiler\\recipes.yaml', 'DATA')],
               strip=None,
               upx=True,
               name='dolo-matlab')