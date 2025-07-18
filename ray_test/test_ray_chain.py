import ray


@ray.remote
class ChainActor:
    def __init__(self):
        pass

    def get(self, obj: ray.ObjectRef):
        actual = ray.get(obj)
        return actual + " Actor"


@ray.remote
def task1():
    return "Task 1"

@ray.remote
def task2():
    return task1.remote()


@ray.remote
class Foo:
    def __init__(self):
        print("Foo")

    def print(self, actor):
        print('I am in Foo', actor)
        actor.print.remote()

@ray.remote
class Bar:
    def __init__(self):
        print("Bar")

    def print(self):
        print('I am in Bar')

def main():
    ray.init()
    actor = ChainActor.remote()
    res = actor.get.remote(task2.remote())
    print(ray.get(res))

    foo = Foo.remote()
    bar = Bar.remote()
    foo.print.remote(bar)

if __name__ == "__main__":
    main()