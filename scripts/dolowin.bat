set "pyinstallerpath=d:\Documents\Python\pyinstaller-1.5.1\"
set "makespec=%pyinstallerpath%Makespec.py"
set "build=%pyinstallerpath%Build.py"

python %makespec% --onefile ../src/bin/dolo-recs.py
python %build% .\dolo-recs.spec
