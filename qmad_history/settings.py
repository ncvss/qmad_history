# import os

# this_dir = os.path.dirname(os.path.curdir)
# settings_file = os.path.join(this_dir, "qmad_history", "settings.txt")

# capab = dict()

# # take the settings from the text file
# with open(settings_file, "r") as sfile:
#     data = sfile.readlines()
#     for li in data:
#         wo = li.split()
#         capab[wo[0]] = bool(wo[1])

capab = {"parallelise": False, "vectorise": False}
