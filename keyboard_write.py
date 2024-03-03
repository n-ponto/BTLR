from pynput.mouse import Listener
import threading
import keyboard
import pyautogui
from time import sleep

DELAY = 0.01
LONG_DELAY = 0.4

SOURCE_FILE = 'keyboard.txt'
SOURCE_FILE = r'C:\Users\noah\repos\BTLR\wake\audio_collection\sample_collector.py'

continue_running = True


def simulate():
    # Read from the keyboard file
    def read_keyboard_file():
        with open(SOURCE_FILE, 'r') as file:
            return file.read().strip()

    txt = read_keyboard_file()
    keyboard.press_and_release('alt + tab')
    sleep(0.5)

    # For each line in the input get the number of tabs
    prev_tabs = 0
    prev_line = ''
    inside_comment = False
    for line in txt.split('\n'):
        if not continue_running:
            exit()
        # Get the number of tabs at the beginning
        num_tabs = len(line) - len(line.lstrip('\t'))
        num_spaces = len(line) - len(line.lstrip())
        num_tabs += num_spaces // 4
        line = line.lstrip()

        # Check for special termination cases
        if prev_line.startswith('return') or prev_line.startswith('raise') or any([prev_line == word for word in ['pass', 'break', 'continue']]): 
            prev_tabs -= 1

        comment_start = False
        if not inside_comment and line.startswith("\"\"\""):
            comment_start = True
            line = line.lstrip("\"")

        comment_end = False
        # Check if there's an ending comment
        if line.endswith("\"\"\""):
            line = line.rstrip('\"')
            inside_comment = False
            comment_end = True

        add_tabs = 0
        if not inside_comment and prev_line.endswith(':'):
            print('not changing tab')
            pass
        elif prev_line.count('(') > prev_line.count(')'):
            print('skipping bc no closing )')
            pass
        # If the number of tabs is greater than the previous number of tabs
        elif num_tabs > prev_tabs:
            # Add the difference to the line
            add_tabs = num_tabs - prev_tabs
        # Else if the number of tabs is less than the previous number of tabs
        elif num_tabs < prev_tabs:
            # Add backspaces to the line
            [keyboard.press_and_release("backspace")
             for _ in range(prev_tabs - num_tabs)]

        # Add the tabs
        [keyboard.press_and_release("tab") for _ in range(add_tabs)]

        # Print starting comment
        if comment_start:
            sleep(DELAY)
            print('Starting comment')
            pyautogui.write("\"\"\"", DELAY)
            sleep(LONG_DELAY)

        # Print the line
        sleep(DELAY)
        pyautogui.write(line, DELAY)
        print("\""+line+"\"", '\t', num_tabs, add_tabs, prev_tabs, inside_comment)

        if comment_end:
            sleep(DELAY)
            keyboard.press_and_release('ctrl + right')

        sleep(DELAY)
        keyboard.press_and_release('enter')
        sleep(LONG_DELAY)
        print('enter')

        if inside_comment and len(line) < 1:
            # Don't update indent when in a comment for blank lines
            continue
        prev_tabs = num_tabs
        prev_line = line

        if comment_start and not comment_end:
            inside_comment = True


# Start simulate in another thread
continue_running = True
write_thread = threading.Thread(target=simulate, daemon=True)


def on_click(x, y, button, pressed):
    global continue_running
    continue_running = False
    write_thread.join()
    print('Thread finished')
    exit()


listener = Listener(on_click=on_click)
listener.start()
print('Started listener')
write_thread.start()
print('Started writing')

while (continue_running):
    sleep(0.1)

listener.stop()
