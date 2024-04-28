import subprocess


def get_top_left_corner_of_active_app(app_name):
    script = f"""
tell application "System Events"
    set frontApp to first application process whose name is "{app_name}"
    try
        set frontWindow to window 1 of frontApp
        set windowPosition to position of frontWindow
        return windowPosition
    on error
        return "Window position not available"
    end try
end tell"""
    return subprocess.check_output(["osascript", "-e", script]).decode("utf-8").strip()


def get_active_window_name():
    script = 'tell application "System Events" to get the name of the first process whose frontmost is true'
    return subprocess.check_output(["osascript", "-e", script]).decode("utf-8").strip()


def send_notification(title, text):
    osascript_command = f"""
    display dialog "{text}" with title "{title}" buttons {{"Logs", "Thanks, Coach! Back to work."}} default button 2
    if the button returned of the result is "Logs" then
        do shell script "open coaching_responses.txt && tail -n +1 coaching_responses.txt"
    end if
    """
    subprocess.run(["osascript", "-e", osascript_command])
