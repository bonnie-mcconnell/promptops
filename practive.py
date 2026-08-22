from contextlib import contextmanager

@contextmanager
def resource():
    print("SETUP")
    try:
        yield "the value"
    finally:
        print("CLEANUP")

with resource() as r:
    print("using", r)
    raise ValueError("smth broke")