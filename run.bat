@echo off
echo Running Medical LLM Semantic Communication Pipeline...
python test.py %*
if %ERRORLEVEL% NEQ 0 (
  py test.py %*
)
