import glob
from PIL import Image


def dif_sort(list_of_string):
    
def make_gif(frame_folder):
    files = glob.glob(f"{frame_folder}/*.png")
    files.sort()
    frames = [Image.open(image) for image in files]
    print(files)
    frame_one = frames[0]
    frame_one.save("e28.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)
    
if __name__ == "__main__":
    make_gif("./gif_folder/")