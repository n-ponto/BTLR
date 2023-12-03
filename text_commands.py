from command_handling import CommandHandler

if __name__ == "__main__":
    import utils
    command_handler = CommandHandler(None, sample_size=16000, sample_rate=2, is_pi=utils.is_pi())
    while True:
        command = input('Enter a command: ')
        command_handler.handle(command)