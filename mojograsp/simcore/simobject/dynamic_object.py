import pybullet as p
from . import objectbase


class DynamicObject(objectbase.ObjectBase):
    def __init__(self, filename, fixed=False, base_pos=None, base_orn=None):
        super().__init__(filename, fixed, base_pos, base_orn)


if __name__ == '__main__':
    pass
