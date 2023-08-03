@echo off
chcp 65001>nul

if not defined STREAMLIT (set STREAMLIT=streamlit)

%STREAMLIT% run 1_🦊_Homepage.py%*

pause