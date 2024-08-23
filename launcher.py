import os
import sys
import win32com.client

def create_shortcut():
    try:
        # Path to the Python executable
        python_exe = sys.executable.replace("python.exe", "pythonw.exe")

        # Path to your PyQt5 GUI script
        script_path = r'C:\Users\syazw\Desktop\UMP\sem5\BTI3423 MACHINE VISION\Project\Project Face Detection & Emotion Recognition\gui.py'

        # Path to the icon file
        icon_path = r'C:\Users\syazw\Desktop\UMP\sem5\BTI3423 MACHINE VISION\Project\Project Face Detection & Emotion Recognition\FDER.ico'

        # Path where you want to create the shortcut
        desktop = os.path.join(os.environ['USERPROFILE'], 'Desktop')
        shortcut_path = os.path.join(desktop, 'FDER.lnk')

        # Verify paths
        if not os.path.isfile(python_exe):
            print(f"Python executable not found: {python_exe}")
            return
        if not os.path.isfile(script_path):
            print(f"Python script not found: {script_path}")
            return
        if not os.path.isfile(icon_path):
            print(f"Icon file not found: {icon_path}")
            return

        # Create the shortcut
        shell = win32com.client.Dispatch('WScript.Shell')
        shortcut = shell.CreateShortcut(shortcut_path)
        shortcut.TargetPath = python_exe
        shortcut.Arguments = f'"{script_path}"'
        shortcut.IconLocation = icon_path
        shortcut.WorkingDirectory = os.path.dirname(script_path)
        shortcut.save()

        print(f'Shortcut created at {shortcut_path}')
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    create_shortcut()
