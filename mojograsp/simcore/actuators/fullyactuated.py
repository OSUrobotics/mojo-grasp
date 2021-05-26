import pybullet as p


class FullyActuated():
	def __init__(self, joint_index, unique_id):
		self.joint_index = joint_index
		self.id = unique_id
		print("setting joints")

	#gets joint index
	def get_joint_index(self):
		return self.joint_index

	#gets joint names from index
	def get_joint_index_names(self):
		return list(self.joint_index.keys())

	#gets joint numbers from index
	def get_joint_index_numbers(self):
		return list(self.joint_index.values())

	#takes a list of names and returns the joint numbers 
	def get_joints_from_name(self, names):
		joint_num_list = []
		for i in names:
			if(i in self.joint_index):
				joint_num_list.append(self.joint_index[i])
			else:
				print("Could not find " + str(i) + " in joint index")

		return joint_num_list 

	#gets joint names from a list of joint numbers
	#very inneficient for large lists, more of a debugging feature
	def get_joints_from_number(self, numbers):
		joint_name_list = []
		
		name = list(self.joint_index.keys())
		num = list(self.joint_index.values())

		for i in numbers:
			found = False
			for j in range(len(num)):
				if(i == num[j]):
					joint_name_list.append(name[j])
					found = True
					break

			if(found == False):
				print("Could not find joint " + str(i) + " in joint index")
		
		return joint_name_list

