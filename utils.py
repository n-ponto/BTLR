class ConsoleColors:
    """Class to with ANSI escape codes for colors."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_warning(string: str):
    """Prints warning to console."""
    print(f'{ConsoleColors.WARNING}[WARNING] {string}{ConsoleColors.ENDC}')


def print_error(string: str):
    print(f'{ConsoleColors.FAIL}[ERROR] {string}{ConsoleColors.ENDC}')


def is_pi():
    """
    Check if running on a raspberry pi.
    Returns True if running on a raspberry pi, False otherwise.
    """
    import os
    return os.name != 'nt'


if __name__ == "__main__":
    print(f"{ConsoleColors.HEADER} HEADER {ConsoleColors.ENDC}")
    print(f"{ConsoleColors.OKBLUE} OKBLUE {ConsoleColors.ENDC}")
    print(f"{ConsoleColors.OKCYAN} OKCYAN {ConsoleColors.ENDC}")
    print(f"{ConsoleColors.OKGREEN} OKGREEN {ConsoleColors.ENDC}")
    print(f"{ConsoleColors.WARNING} WARNING {ConsoleColors.ENDC}")
    print(f"{ConsoleColors.FAIL} FAIL {ConsoleColors.ENDC}")
    print(f"{ConsoleColors.ENDC} ENDC {ConsoleColors.ENDC}")
    print(f"{ConsoleColors.BOLD} BOLD {ConsoleColors.ENDC}")
    print(f"{ConsoleColors.UNDERLINE} UNDERLINE {ConsoleColors.ENDC}")

    print_warning("This is a warning")
    print_error("This is an error")
    print(f'is_pi = {is_pi()}')
