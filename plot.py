import matplotlib.pyplot as plt
import torch
from utils.plot import plot0, plot1, plot2, plot3, load, calc
from matplotlib import cm
import matplotlib
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['patch.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

base_file_path = "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.base"

file_paths = [
    #"/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.base",
    "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.A",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.A",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.B",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.B",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.B"]

# calc(file_paths, base_file_path)
# # stats = data['over_zero'] / data['n']


# # plot0(file_paths, title="Ratio_of_activated_neurons")
# file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.B"]
# calc(file_paths,base_file_path)

# file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.A",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.B"]

# calc(file_paths,base_file_path)

# file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.A",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.B"]

# calc(file_paths,base_file_path)

# file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.A"]

# calc(file_paths,base_file_path)

# file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.B"]

# calc(file_paths,base_file_path)

# file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.B",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.A"]

# calc(file_paths,base_file_path)

# file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.B",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.B"]

# calc(file_paths,base_file_path)
# file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.A"]

# calc(file_paths,base_file_path)

# file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.B"]
# calc(file_paths,base_file_path)
# file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.B",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.A"]


# calc(file_paths,base_file_path)

# file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.B",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.B"]


# calc(file_paths,base_file_path)

labels = ["Expertise_A or Informativeness_A",
          "Expertise_Informativeness_AA",
          "Expertise_A",
          "Informativeness_A"]
# plot2(base_file_path, file_paths, title="Expertise_A_Informativeness_A", labels=labels)

file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.B",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.B",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise_Informativeness.BB"]

labels = ["Expertise_B or Informativeness_B",
          "Expertise_Informativeness_BB",
          "Expertise_B",
          "Informativeness_B"]



file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.A",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise_Informativeness.AA"]

labels = ["Expertise_A or Informativeness_A",
          "Expertise_Informativeness_AA",
          "Expertise_A",
          "Informativeness_A"]
# plot2(base_file_path, file_paths, title="Expertise_A_Informativeness_A", labels=labels)

file_paths = [
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.B",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.A",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness_Style.BA"]

labels = ["Informativeness_B or Style_A",
          "Informativeness_Style_BA",
          "Informativeness_B",
          "Style_A",
          ]
# plot2(base_file_path, file_paths, title="Informativeness_B_Style_A", labels=labels)


file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.B",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.B",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise_Style.BB"]

labels = ["Expertise_B or Style_B",
          "Expertise_Style_BB",
          "Expertise_B",
          "Style_B"]
# plot2(base_file_path, file_paths, title="Expertise_B_Style_B", labels=labels)

file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.A",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.A",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise_Informativeness_Style.AAA"]

labels = ["Expertise_A",
          "Informativeness_A",
          "Style_A", 
          "Expertise_A or Informativeness_A or Style_A",
          "Expertise_Informativeness_Style_AAA"]

plot3(base_file_path, file_paths, title="Expertise_A_Informativeness_A_Style_A", labels=labels)

