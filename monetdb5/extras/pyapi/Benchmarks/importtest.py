


import sys
arguments = sys.argv
if (len(arguments) <= 1):
    print("No arguments provided.")
    quit()


library = arguments[1]
try:
    i = __import__(library)
except:
    sys.exit(1)

