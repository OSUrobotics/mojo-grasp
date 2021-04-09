import os

class URDF_Generator():
    """
    Helps user to create a URDF file
    """
    def __init__(self, file_location):
        self.urdf_text = ''
        self.dir = file_location

    def listConverter(self, pose):
        """
        converts a list into a string to be used in the urdf file
        """
        lis = ""
        for i in range(len(pose)):
            if i < len(pose) - 1:
                lis += str(pose[i]) + " "
            else:
                lis += str(pose[i])
        return lis

    def new_urdf(self):
        """
        makes the urdf file blank for a new file to be created
        """
        self.urdf_text = ''

    def start_file(self, gripper_name='default'):
        """
        called for starting a urdf, adding the initial info for the urdf file
        Input: gripper_name: name of the gripper the file is associated with
        """
        self.urdf_text += f"""<?xml version="1.0" ?>
<robot name="{gripper_name}">"""

    def end_file(self):
        """
        adds the ending info of a urdf file
        """
        self.urdf_text += """

</robot>"""


    def link(self, name="", pose=(0, 0, 0, 0, 0, 0), scale=(1, 1, 1), mass=0.5, model_name=''):
        """
        adds a link to the urdf file
        Inputs: name: name of the link
                pose: (x,y,z,r,p,y) of the link relative to the origin of it's parent link, typically this is taken
                    care of by the joint not the link
                scale: (x,y,z) scale factor in each direction
                mass: not currently used
                model_name: use if model name is different then link name
        """

        xyz = self.listConverter(pose[:3])
        rpy = self.listConverter(pose[3:])
        scale = self.listConverter(scale)

        if model_name == '':
            model_name = name

        col_loc = 'obj_files/' + model_name + '_collision.obj'
        vis_loc = 'obj_files/' + model_name + '.obj'

        self.urdf_text += f"""

    <link name="{name}">
        <visual>
            <origin rpy="{rpy}" xyz="{xyz}"/>
            <geometry>
                <mesh filename="{vis_loc}" scale="{scale}"/>
            </geometry>
        </visual>
        <collision>
            <geometry>
                <mesh filename="{col_loc}" scale="{scale}"/>
            </geometry>
        </collision>
    </link>"""
    # figure out if this is needed as of now no
    # <inertial>
    #         <mass value="{mass}"/>
    #         <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    #     </inertial>

    def joint(self, name, Type, child, parent, axis=(0,0,0), rpy_in=(0, 0, 0), xyz_in=(0.0, 0.0, 0.0)):
        """
        adds a joint to the urdf file
        Inputs: name: joint name
                Type: revolute, or fix for now more can be added later
                child: name of child link
                parent: name of parent link
                axis: (x,y,z) axis of rotation
                rpy_in: (r,p,y) rotation of child link relative to parent
                xyz_in: (x,y,z) location of child's origin relative to parents origin
        """

        axis = self.listConverter(axis)
        xyz = self.listConverter(xyz_in)
        rpy = self.listConverter(rpy_in)
        
        self.urdf_text += f"""

    <joint name="{name}" type="{Type}">
        <parent link="{parent}"/>
        <child link="{child}"/>
        <origin rpy="{rpy}" xyz="{xyz}"/>
        <axis xyz="{axis}"/>
        <limit effort="1000.0" lower="-3.14" upper="3.14" velocity="0.5"/>
    </joint>"""

    def write(self, filename='urdf_Test'):
        """
        Once all components for the urdf file are complete write the file
        Inputs: filename: name of the urdf file
        """
        directory = self.dir
        content = self.urdf_text
        with open(f'{directory}{filename}.urdf', 'w') as file:
            file.write(content)
            file.close()


if __name__ == '__main__':
    
    direct = os.path.dirname(__file__)
    robot = URDF_Generator(file_location=direct)

    # test file
    # robot.start_file('testing')
    # robot.link('body_palm_lpinvrpin',(0, 0, 0, 1.57, 0, 0))
    # robot.joint('left_prox_joint', "continuous", "body_l_prox_pinvpin2", "body_palm_lpinvrpin", (1, 0, 0))
    # robot.link('body_l_prox_pinvpin2', (0, 0, 0, 1.57, 0, 0))
    # robot.end_file()
    # robot.write('URDF_test')
