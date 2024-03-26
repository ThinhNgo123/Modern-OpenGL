from OpenGL.GL import glGetError, glDebugMessageCallback

def debug_gl(func):
    # def inner(*args, **kwargs):
    #     try:
    #         return func(*args, **kwargs)
    #     except:
    #         print("[Error]:", glGetError())
    # return inner
    pass

def debug_callback(*args):
    for arg in args:
        print(arg)