import pybullet as p
import numpy as np
from mojograsp.simcore.jacobianIK import matrix_helper as mh
import time


class ForwardKinematicsSIM():
    def __init__(self, hand_id, finger_info: dict) -> None:
        # finger information and pybullet body id
        self.hand_id = hand_id
        self.finger_name = finger_info["name"]
        self.num_links = finger_info["num_links"]
        self.link_lengths = finger_info["link_lengths"]
        # transforms
        self.link_translations = []
        self.link_rotations = []
        self.link_rotations_original = []
        self.current_angles = []
        # Current pose of each link
        self.current_poses = []
        self.link_ids = []
        self.cnt = 0
        # ee_ending location
        self.original_ee_end = None
       # Create our link id and name lists above
        self.initialize_transforms()
        # Holds the link names for each finger for debugging purposes
        self.debug_id_new = None

    def get_link_ids(self):
        # Gets all link ids for the given finger_name, finger name must be in the link name
        for i in range(p.getNumJoints(self.hand_id)):
            j_info = p.getJointInfo(self.hand_id, i)
            j_name = j_info[12].decode('UTF-8')
            j_index = j_info[0]
            # print('JNAMES',j_name)
            # get all links that contain finger name, except for those with static in the name
            if self.finger_name in j_name and "static" not in j_name and "sensor" not in j_name:
                self.link_ids.append(j_index)

    def update_ee_end_point(self, ee_ending_local=None):
        if ee_ending_local is not None:
            self.link_lengths[-1] = ee_ending_local
        else:
            self.link_lengths[-1] = self.original_ee_end

    def update_poses_from_sim(self):
        # get each current link pose in global coordinates [(x,y,z), (x,y,z,w)]
        for id in self.link_ids:
            l_state = p.getLinkState(self.hand_id, id)
            self.current_poses.append([list(l_state[0]), list(l_state[1])])

    def update_angles_from_sim(self):
        # get each current link pose in global coordinates [(x,y,z), (x,y,z,w)]
        for i in range(len(self.current_angles)):
            self.current_angles[i] = p.getJointState(self.hand_id, self.link_ids[i])[0]
        self.set_joint_angles(self.current_angles)

    def initialize_transforms(self):
        # get initial poses from sim and link ids (MUST BE DONE IN THIS ORDER)
        self.get_link_ids()
        self.update_poses_from_sim()
        

            # get the base link out first
        base_link_t = mh.create_translation_matrix(self.current_poses[0][0])
        base_link_r = mh.create_rotation_matrix(p.getEulerFromQuaternion(self.current_poses[0][1])[2])
        self.link_translations.append(base_link_t)
        self.link_rotations.append(base_link_r)
        self.link_rotations_original.append(base_link_r)
        self.current_angles.append(p.getJointState(self.hand_id, self.link_ids[0])[0])
        # Get the transformation from previous link to next link
        for i in range(1, len(self.current_poses)):
            # print(self.current_poses[i][0])
            mat_t = mh.create_translation_matrix(self.current_poses[i][0])
            mat_r = mh.create_rotation_matrix(p.getEulerFromQuaternion(self.current_poses[i][1])[2])
            # Since poses are in the global frame we need to find the matrix that takes us from previous to next using A@B.I
            #mat_t_link = mat_t @ np.linalg.inv(self.link_translations[-1])
            mat_t_inv = mh.create_translation_matrix(self.current_poses[i-1][0])
            mat_t_link = mat_t @ np.linalg.inv(mat_t_inv)
            mat_r_link = mat_r @ np.linalg.inv(self.link_rotations[-1])
            self.link_translations.append(mat_t_link)
            self.link_rotations.append(mat_r_link)
            self.link_rotations_original.append(mat_r_link)
            self.current_angles.append(p.getJointState(self.hand_id, self.link_ids[i])[0])
        self.original_ee_end = self.link_lengths[-1]
        self.link_rotations_original.reverse()

        # DEBUG  PRINTOUTS
        # print("DEBUG F1 STARTING Translation: ", self.link_translations)
        # print("DEBUG F1 STARTING Rotations: ", self.link_rotations)
        # print("DEBUG F1 STARTING Angles: ", self.current_angles)
        # print("DEBUG F1 END EFFECTOR: ", self.original_ee_end)


    # def initialize_transforms(self):
    #     # get initial poses from sim and link ids (MUST BE DONE IN THIS ORDER)
    #     self.get_link_ids()
    #     self.update_poses_from_sim()

    #     # get the base link out first
    #     base_link_t = mh.create_translation_matrix(self.current_poses[0][0])
    #     base_link_r = mh.create_rotation_matrix(p.getEulerFromQuaternion(self.current_poses[0][1])[2])
    #     self.link_translations.append(base_link_t)
    #     self.link_rotations.append(base_link_r)
    #     self.link_rotations_original.append(base_link_r)
    #     self.current_angles.append(p.getJointState(self.hand_id, self.link_ids[0])[0])
    #     # Get the transformation from previous link to next link
    #     for i in range(1, len(self.current_poses)):
    #         print(self.current_poses[i][0])
    #         mat_t = mh.create_translation_matrix(self.current_poses[i][0])
    #         mat_r = mh.create_rotation_matrix(p.getEulerFromQuaternion(self.current_poses[i][1])[2])
    #         # Since poses are in the global frame we need to find the matrix that takes us from previous to next using A@B.I
    #         #mat_t_link = mat_t @ np.linalg.inv(self.link_translations[-1])
    #         mat_t_inv = mh.create_translation_matrix(self.current_poses[i-1][0])
    #         mat_t_link = mat_t @ np.linalg.inv(mat_t_inv)
    #         mat_r_link = mat_r @ np.linalg.inv(self.link_rotations[-1])
    #         self.link_translations.append(mat_t_link)
    #         self.link_rotations.append(mat_r_link)
    #         self.link_rotations_original.append(mat_r_link)
    #         self.current_angles.append(p.getJointState(self.hand_id, self.link_ids[i])[0])
    #     self.original_ee_end = self.link_lengths[-1]
    #     self.link_rotations_original.reverse()
    #     # DEBUG  PRINTOUTS
    #     #print("DEBUG F1 STARTING Translation: ", self.link_translations)
    #     #print("DEBUG F1 STARTING Rotations: ", self.link_rotations)
    #     #print("DEBUG F1 STARTING Angles: ", self.current_angles)

    def set_joint_angles(self, angles):
        # sets the joint angles and updates rotation matrices
        for i in range(len(angles)):
            mat_r = mh.create_rotation_matrix(angles[i])
            self.link_rotations[i] = mat_r
            self.current_angles[i] = angles[i]

    def calculate_forward_kinematics(self):
        debug = []
        link_location = np.identity(3)
        # iterate over every link and multiply the transforms together
        for i in range(len(self.link_lengths)):
            link_location = link_location @ self.link_translations[i] @ self.link_rotations[i]
            # debug adding link locations
            debug.append(link_location @ [0, 0, 1])
            #print(f"LINK {i}: {link_location}")
        # finally get the end effector end location
        link_end = mh.create_translation_matrix(self.link_lengths[-1])
        link_location = link_location @ link_end
        # debug adding link locations
        debug.append(link_location @ [0, 0, 1])
        # self.cnt += 1
        # if self.cnt == 20:
        #     self.cnt = 0
        #     self.debug_show_link_positions(debug)
        #     time.sleep(1)
        return link_location @ [0, 0, 1]

    def debug_show_link_positions(self, points):
        # temporary debug function to show links and compare
        if self.debug_id_new:
            p.removeUserDebugItem(self.debug_id_new)
        for i in points:
            i[2] = .05
        self.debug_id_new = p.addUserDebugPoints(
            points,
            [[255, 0, 0]] * len(points),
            pointSize=10)
