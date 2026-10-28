@echo off
title Bionic Lite 服务启动器
cd /d D:\code\Marvis\bionic_lite

netstat -ano | findstr :1235 >nul 2>&1
if %errorlevel%==0 (
  echo [信息] 服务已在运行，直接打开页面...
) else (
  echo [信息] 正在启动 Bionic Lite 服务（端口 1235）...
  start "" "D:\Program Files\Tencent\Marvis\MarvisAgent\1.60.2900.698\runtime\python311\python.exe" bionic_lite.py serve --config config.json
  timeout /t 2 /nobreak >nul
)

start http://192.168.0.133:1235/
echo [信息] 已打开页面: http://192.168.0.133:1235/
pause
