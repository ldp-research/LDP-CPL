
'''
    Utility functions
'''

def list_to_string(l):
    s = ""
    for i in l:
        s += str(i) + " "
    return s

def string_to_list(s):
    __l = s.split(" ")
    l = []
    for i in __l:
        if i != "" :
            l.append((i))
    return l
