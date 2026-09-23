@echo off
echo Running Medical LLM Semantic Communication Pipeline...
py -3.11 test.py %*
if %ERRORLEVEL% NEQ 0 (
  "C:\Users\Dwaragesh k\AppData\Local\Programs\Python\Python311\python.exe" test.py %*
)
