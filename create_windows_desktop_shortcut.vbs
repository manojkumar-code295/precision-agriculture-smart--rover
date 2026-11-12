' =========================================================================
' VBScript to create "Plant Disease Detection AI" desktop shortcut on Windows
' Double-click this script once on Windows to place the shortcut on your Desktop
' =========================================================================

Set WshShell = CreateObject("WScript.Shell")
strDesktop = WshShell.SpecialFolders("Desktop")
strCurrentDir = WshShell.CurrentDirectory

Set oShortcut = WshShell.CreateShortcut(strDesktop & "\Plant Disease Detection AI.lnk")
oShortcut.TargetPath = strCurrentDir & "\Plant Disease Detection AI.bat"
oShortcut.WorkingDirectory = strCurrentDir
oShortcut.Description = "Plant Disease Detection AI - Desktop Camera Tester"
oShortcut.WindowStyle = 1
oShortcut.IconLocation = "shell32.dll,14"
oShortcut.Save

WScript.Echo "Shortcut created successfully on your Windows Desktop: Plant Disease Detection AI"
