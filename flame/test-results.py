#! -*- coding: utf-8 -*-

class results:
    def __init__(self):
        self.d = {}

    def addValue(self,x,y):
        self.d[x] = y

def do_something (r,a,b):

    r.addValue(a,b)


my_result = results ()
print (my_result.d)

do_something(my_result,'primero',1)
print (my_result.d)

do_something(my_result,'segundo',2)
print (my_result.d)



