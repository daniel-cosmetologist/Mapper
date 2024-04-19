import sys

class LogMobileDistinguisher:
    def __init__(self, desktop_file_name, mobile_file_name):
        self.desktop = self.read_file(desktop_file_name)
        self.mobile = self.read_file(mobile_file_name)
        self.process_log()

    def read_file(self, file_name: str) -> set:
        with open(file_name, 'r') as fi:
            return {line.strip() for line in fi}

    def get_line_client(self, line_ua: str) -> str:
        if line_ua in self.desktop:
            return 'desktop'
        elif line_ua in self.mobile:
            return 'mobile'
        return 'unknown'

    def get_line_fields(self, line: str) -> list:
        result = []
        in_string = False
        s = ''
        for field in line.strip().split(' '):
            if not in_string:
                if field.startswith('"') and not field.endswith('"'):
                    in_string = True
                    s = field[1:]
                else:
                    result.append(field)
            else:
                if field.endswith('"'):
                    s += ' ' + field[:-1]
                    result.append(s)
                    in_string = False
                else:
                    s += ' ' + field
        return result

    def process_log(self):
        for line in sys.stdin:
            fields = self.get_line_fields(line)
            # Проверяем, что индекс для User Agent корректен (например, последний элемент в списке)
            client = self.get_line_client(fields[-1])
            print(client + ' ' + line.strip())


if __name__ == '__main__':
    LogMobileDistinguisher('d.txt', 'm.txt')
