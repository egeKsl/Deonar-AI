@echo off
REM ---------- CONFIG - update these if your paths differ ----------
set MEDIAMTX_DIR=C:\rtsp-server
set FFMPEG_BIN=C:\ffmpeg\bin\ffmpeg.exe
set INPUT=C:\Users\ubada\Downloads\test_163.mp4
set RTMP_URL=rtmp://127.0.0.1:1935/mystream
set RTSP_URL=rtsp://127.0.0.1:8554/mystream
REM ---------------------------------------------------------------

echo Killing any existing ffmpeg publishers...
tasklist /FI "IMAGENAME eq ffmpeg.exe" | find /I "ffmpeg.exe" >nul
if %errorlevel%==0 (
    taskkill /IM ffmpeg.exe /F >nul 2>&1
    timeout /t 1 /nobreak >nul
)

echo Starting mediamtx server (new window)...
start "mediamtx" cmd /k "cd /d %MEDIAMTX_DIR% && mediamtx.exe"

echo Waiting 2 seconds for mediamtx to initialize...
timeout /t 2 /nobreak >nul

echo Starting ffmpeg publisher with downscale, bitrate limit and silent AAC audio (new window)...
start "publisher" cmd /k ""%FFMPEG_BIN%" -re -stream_loop -1 -i "%INPUT%" -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 -map 0:v:0 -map 1:a:0 -vf scale=1280:720 -c:v libx264 -preset veryfast -tune zerolatency -b:v 2500k -c:a aac -ac 2 -b:a 128k -shortest -f flv %RTMP_URL%"

echo.
echo ✅ Started. Use this RTSP URL in your pipeline/player:
echo     %RTSP_URL%
echo.
echo Open the mediamtx window to check logs. It should say:
echo     "is publishing ... 2 tracks (H264, AAC)"
pause
