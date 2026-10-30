with open('/proc/1/cmdline') as f:
    print(f.read().replace('\x00', ' '))
