import pybullet as p


class ObjectBase:

    def __init__(self, filename, fixed, base_pos=None, base_orn=None):
        print("init objectbase")

        # contains base variables for all objects
        self.model_file = filename
        self.id = None
        self.origin = None
        self.position = None
        self.orientation = None
        self.obj_id = None
        self.fixed = fixed
        self.start_pos = {}
        self.base_pos = base_pos
        self.base_orn = base_orn
        if 'sdf' in filename:
            print("This is an sdf file")
            self.load_object_sdf()
        else:
            print("This is a urdf file")
            self.load_object()

    # loads object into pybullet, needs to be changed to support urdf, sdf and others right now only supports urdf with
    # no arguments other than fixed
    def load_object(self):
        if self.base_pos is not None and self.base_orn is not None:
            self.base_orn = p.getQuaternionFromEuler(self.base_orn)
            self.id = p.loadURDF(self.model_file, useFixedBase=self.fixed, basePosition=self.base_pos,
                                 baseOrientation=self.base_orn)
        elif self.base_pos is None and self.base_orn is not None:
            self.base_orn = p.getQuaternionFromEuler(self.base_orn)
            self.id = p.loadURDF(self.model_file, useFixedBase=self.fixed, baseOrientation=self.base_orn)
        elif self.base_orn is None and self.base_pos is not None:
            self.id = p.loadURDF(self.model_file, useFixedBase=self.fixed, basePosition=self.base_pos)
        else:
            self.id = p.loadURDF(self.model_file, useFixedBase=self.fixed)

        start_pos, start_orn = self.get_curr_pose(self.id)
        self.start_pos.update({self.id: [start_pos, start_orn]})
        print("loading object")

    def load_object_sdf(self):
        # self.id = p.loadURDF(self.model_file, useFixedBase=self.fixed)
        ids = p.loadSDF(self.model_file)
        for i in range(len(ids)):
            if i == 0:
                self.id = ids[i]
            else:
                self.obj_id = ids[i]
            start_pos, start_orn = self.get_curr_pose(ids[i])
            self.start_pos.update({ids[i]: [start_pos, start_orn]})
        print("loading objects")

    @staticmethod
    def get_curr_pose(object_id):
        curr_pos, curr_orn = p.getBasePositionAndOrientation(object_id)
        return curr_pos, curr_orn

    def get_dimensions(self):
        return p.getVisualShapeData(self.id)[0][3]
